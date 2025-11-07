# freeway_llm_guided_film.py
# PPO on HackAtari Freeway with LLM-guided attention + FiLM/Adapter conditioning.
# LLM provider: DeepSeek (chat completions). Falls back to heuristic if not configured.
#
# Requirements:
#   pip install hackatari gymnasium stable-baselines3 torch numpy
#   (optional) pip install requests  # for direct HTTP call to DeepSeek
#
# Set DeepSeek API key (optional) to enable LLM:
#   export DEEPSEEK_API_KEY="sk-..."        # Linux/Mac
#   set DEEPSEEK_API_KEY=sk-...            # Windows
#
# Run:
#   python freeway_llm_guided_film.py

import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from local_deepseek import LocalDeepSeek

from hackatari.core import HackAtari
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# -------------------- Env wrapper: add object slots --------------------
class AddObjectsWrapper(gym.ObservationWrapper):
    """
    Dict obs: {"pixels": HxWxC uint8, "objects": (N,5) float32 [x,y,w,h,type_id]}
    """
    def __init__(self, env, max_slots=12):
        super().__init__(env)
        self.max_slots = max_slots
        self.obj_dim = 5
        pix_space = env.observation_space  # Box(0..255, (210,160,3), uint8)
        obj_space = spaces.Box(low=0.0, high=255.0, shape=(self.max_slots, self.obj_dim), dtype=np.float32)
        self.observation_space = spaces.Dict({"pixels": pix_space, "objects": obj_space})

    def _extract_objects(self):
        slots = []
        for o in getattr(self.env, "objects", []):
            x = float(getattr(o, "x", 0)); y = float(getattr(o, "y", 0))
            w = float(getattr(o, "w", 0)); h = float(getattr(o, "h", 0))
            name = getattr(o, "category", type(o).__name__).lower()
            if w <= 0 or h <= 0: 
                continue
            if "player" in name or "chicken" in name: tid = 1
            elif "car" in name or "vehicle" in name or "truck" in name: tid = 2
            elif "goal" in name or "flag" in name or "finish" in name: tid = 3
            else: tid = 0
            slots.append([x, y, w, h, tid])
        if len(slots) < self.max_slots:
            slots += [[0,0,0,0,0]] * (self.max_slots - len(slots))
        else:
            slots = slots[:self.max_slots]
        return np.array(slots, dtype=np.float32)

    def observation(self, obs):
        return {"pixels": obs, "objects": self._extract_objects()}

# -------------------- DeepSeek LLM client --------------------
class DeepSeekClient:
    """
    Minimal DeepSeek chat client via HTTP.
    Set DEEPSEEK_API_KEY. Endpoint/model can be customized with env vars.
    """
    def __init__(self, model=None, base_url=None, timeout=10):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.timeout = timeout
        self.enabled = bool(self.api_key)
        try:
            import requests  # noqa
            self._requests_ok = True
        except Exception:
            self._requests_ok = False
            self.enabled = False

    def guidance(self, obj_list):
        """
        Call DeepSeek with a compact prompt and ask for JSON:
          { "alpha": [..N..], "edges": [[i,j,score], ...], 
            "film": { "img_gamma": [...], "img_beta": [...],
                      "obj_gamma": [...], "obj_beta": [...] } }
        alpha in [-1,1], edges score in [-1,1], FiLM vectors ~ length K each.
        """
        if not self.enabled:
            return None  # triggers heuristic fallback

        import requests
        # Build a concise description
        lines = []
        for i, (x,y,w,h,tid) in enumerate(obj_list):
            lines.append(f"[{i}] tid={int(tid)} xywh=({int(x)},{int(y)},{int(w)},{int(h)})")
        obj_txt = "\n".join(lines) if lines else "(none)"

        prompt = (
            "Task: Atari Freeway. Agent=player(tid=1). Cars tid=2. Goal/finish tid=3.\n"
            "Produce JSON with salience alpha[-1..1] per slot, relation edges (player↔others), "
            "and FiLM params for image & object encoders.\n"
            "Objects:\n" + obj_txt + "\n"
            'Return JSON only, schema: {"alpha":[...],"edges":[[i,j,score],...],'
            '"film":{"img_gamma":[K],"img_beta":[K],"obj_gamma":[K],"obj_beta":[K]}}'
        )

        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Attempt to parse JSON substring
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                js = json.loads(content[start:end+1])
                return js
        except Exception:
            pass
        return None  # fallback

# -------------------- Heuristic fallback guidance --------------------
def heuristic_guidance(obj_slots: th.Tensor, img_channels=64, obj_dim=128):
    """
    Returns alpha (B,N), B (B,N,N), and FiLM vectors for image & object encoders.
    Alpha highlights cars & goal; B discourages attend(player->car) and encourages player->goal.
    FiLM vectors are simple, fixed magnitudes (can be tuned).
    """
    with th.no_grad():
        Bsz, N, D = obj_slots.shape
        tid = obj_slots[..., 4]
        is_car = (tid == 2).float()
        is_goal = (tid == 3).float()
        is_player = (tid == 1).float()

        alpha = 0.6*is_car + 0.3*is_goal + 0.1*is_player
        alpha = 2.0*alpha - 1.0  # [-1,1]

        B = th.zeros(Bsz, N, N, device=obj_slots.device)
        for b in range(Bsz):
            p_idx = th.nonzero(is_player[b], as_tuple=False).flatten().tolist()
            c_idx = th.nonzero(is_car[b], as_tuple=False).flatten().tolist()
            g_idx = th.nonzero(is_goal[b], as_tuple=False).flatten().tolist()
            for p in p_idx:
                for c in c_idx:
                    B[b, p, c] -= 0.8
                for g in g_idx:
                    B[b, p, g] += 0.8

        # FiLM vectors (per-batch, shared across time)
        img_gamma = th.ones(Bsz, img_channels, device=obj_slots.device) * 1.10  # slight gain
        img_beta  = th.zeros(Bsz, img_channels, device=obj_slots.device)
        obj_gamma = th.ones(Bsz, obj_dim, device=obj_slots.device) * 1.05
        obj_beta  = th.zeros(Bsz, obj_dim, device=obj_slots.device)

        return alpha, B, img_gamma, img_beta, obj_gamma, obj_beta

# -------------------- Attention + FiLM modules --------------------
class BiasedSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x, alpha=None, bias=None):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.n_heads, self.d_k).transpose(1,2)
        k = self.k(x).view(B, N, self.n_heads, self.d_k).transpose(1,2)
        v = self.v(x).view(B, N, self.n_heads, self.d_k).transpose(1,2)

        if alpha is not None:
            scale = (1.0 + alpha.unsqueeze(1).unsqueeze(-1))  # (B,1,N,1)
            k = k * scale
            v = v * scale

        logits = (q @ k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if bias is not None:
            logits = logits + bias.unsqueeze(1)
        attn = logits.softmax(dim=-1)
        y = attn @ v
        y = y.transpose(1,2).contiguous().view(B, N, D)
        return self.o(y)

class FiLM(nn.Module):
    """
    Channel-wise FiLM: y = gamma ⊙ x + beta
    gamma,beta come from LLM (per-batch).
    """
    def __init__(self, C):
        super().__init__()
        self.C = C

    def forward(self, x, gamma, beta):
        # x: (B,C,H,W) or (B,C)
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
            beta  = beta.unsqueeze(-1)
        return gamma * x + beta

# -------------------- Features extractor with attention + FiLM --------------------
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ObjectLLMFiLMExtractor(BaseFeaturesExtractor):
    """
    Pixels -> CNN -> FiLM(img)
    Objects -> proj -> biased self-attn (alpha,B) -> FiLM(obj)
    Fuse pooled features -> final feature vector
    """
    def __init__(self, obs_space: spaces.Dict, max_slots=12, d_model=128, out_dim=256,
                 img_channels=64, use_deepseek=True):
        super().__init__(obs_space, features_dim=out_dim)
        self.max_slots = max_slots
        self.d_model = d_model
        self.img_channels = img_channels

        # CNN on pixels (CHW expected inside forward; convert if needed)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, img_channels, 3, stride=1), nn.ReLU()
        )
        self.img_pool = nn.AdaptiveAvgPool2d((1,1))
        self.img_film = FiLM(img_channels)     # FiLM on conv channels
        self.img_proj = nn.Linear(img_channels, d_model)

        # Objects
        self.obj_proj = nn.Linear(5, d_model)
        self.obj_attn = BiasedSelfAttention(d_model=d_model, n_heads=4)
        self.obj_film = FiLM(d_model)

        # Fusion
        self.fuse = nn.Sequential(
            nn.LayerNorm(2*d_model),
            nn.Linear(2*d_model, out_dim), nn.ReLU()
        )

        # LLM client
        # self.deepseek_enabled = use_deepseek and bool(os.getenv("DEEPSEEK_API_KEY", ""))
        self.deepseek_enabled = True
        #self.deepseek = DeepSeekClient() if self.deepseek_enabled else None
        self.deepseek = LocalDeepSeek(model_id="deepseek-ai/DeepSeek-V2-Lite")
        # Small cache to reduce calls (optional): per batch step you could cache by hash of object slots.

    def _call_deepseek(self, objs_np: np.ndarray, device) -> Tuple[th.Tensor,...]:
        """
        Call DeepSeek; parse JSON into (alpha, B, img_gamma, img_beta, obj_gamma, obj_beta).
        Shapes:
          alpha (N), edges (list), film vectors lengths: img_channels, d_model
        """
        js = self.deepseek.guidance(objs_np.tolist())
        if js is None:
            return heuristic_guidance(th.tensor(objs_np, device=device).unsqueeze(0),
                                      img_channels=self.img_channels, obj_dim=self.d_model)
        # Parse safely with fallbacks
        N = objs_np.shape[0]
        alpha = th.zeros(1, N, device=device)
        try:
            a_list = js.get("alpha", [])
            for i in range(min(N, len(a_list))):
                alpha[0, i] = float(max(-1.0, min(1.0, a_list[i])))
        except Exception:
            pass

        B = th.zeros(1, N, N, device=device)
        try:
            for (i,j,score) in js.get("edges", []):
                if 0 <= i < N and 0 <= j < N:
                    B[0, int(i), int(j)] = float(max(-1.0, min(1.0, score)))
        except Exception:
            pass

        img_gamma = th.ones(1, self.img_channels, device=device)
        img_beta  = th.zeros(1, self.img_channels, device=device)
        obj_gamma = th.ones(1, self.d_model, device=device)
        obj_beta  = th.zeros(1, self.d_model, device=device)
        try:
            film = js.get("film", {})
            def to_vec(key, D):
                arr = film.get(key, [])
                v = th.zeros(D, device=device)
                for i in range(min(D, len(arr))):
                    v[i] = float(arr[i])
                return v
            img_gamma[0] = to_vec("img_gamma", self.img_channels)
            img_beta [0] = to_vec("img_beta",  self.img_channels)
            obj_gamma[0] = to_vec("obj_gamma", self.d_model)
            obj_beta [0] = to_vec("obj_beta",  self.d_model)
        except Exception:
            pass

        return alpha, B, img_gamma, img_beta, obj_gamma, obj_beta

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        px = obs["pixels"]  # (B,H,W,C) or (B,C,H,W) depending on vecenv
        if px.dim() != 4:
            raise RuntimeError("pixels must be rank-4")
        # convert to CHW if needed
        if px.shape[1] != 3:
            px = px.permute(0,3,1,2).contiguous()
        px = px.float() / 255.0

        objs = obs["objects"].float()  # (B,N,5)

        # CNN path
        z = self.cnn(px)                     # (B,C,h,w)
        Bsz, C, h, w = z.shape

        # Compute guidance/FiLM per batch element (we’ll share one call per sample)
        img_gamma_list, img_beta_list = [], []
        alpha_list, bias_list = [], []
        obj_gamma_list, obj_beta_list = []

        for b in range(Bsz):
            # CPU numpy for LLM call
            objs_np = objs[b].detach().cpu().numpy()
            if self.deepseek_enabled and self.deepseek is not None:
                alpha_b, B_b, ig, ib, og, ob = self._call_deepseek(objs_np, device=z.device)
            else:
                alpha_b, B_b, ig, ib, og, ob = heuristic_guidance(objs[b].unsqueeze(0),
                                                                  img_channels=self.img_channels,
                                                                  obj_dim=self.d_model)
            alpha_list.append(alpha_b)       # (1,N)
            bias_list.append(B_b)           # (1,N,N)
            img_gamma_list.append(ig)       # (1,C)
            img_beta_list.append(ib)        # (1,C)
            obj_gamma_list.append(og)       # (1,D)
            obj_beta_list.append(ob)        # (1,D)

        alpha = th.cat(alpha_list, dim=0)        # (B,N)
        bias  = th.cat(bias_list, dim=0)         # (B,N,N)
        img_gamma = th.cat(img_gamma_list, dim=0)# (B,C)
        img_beta  = th.cat(img_beta_list, dim=0) # (B,C)
        obj_gamma = th.cat(obj_gamma_list, dim=0)# (B,D)
        obj_beta  = th.cat(obj_beta_list, dim=0) # (B,D)

        # FiLM on image features (adapter-style channel mod)
        z = self.img_film(z, img_gamma, img_beta)  # (B,C,h,w)
        z_img = self.img_pool(z).squeeze(-1).squeeze(-1)  # (B,C)
        z_img = self.img_proj(z_img)                     # (B,D)

        # Object path + biased attention + FiLM
        z_obj = self.obj_proj(objs)                      # (B,N,D)
        z_obj = self.obj_attn(z_obj, alpha=alpha, bias=bias)   # (B,N,D)
        # Apply FiLM to token dimension (treat channels=D)
        z_obj = z_obj.transpose(1,2)                     # (B,D,N)
        z_obj = self.obj_film(z_obj, obj_gamma, obj_beta)      # (B,D,N)
        z_obj = z_obj.transpose(1,2).mean(dim=1)         # (B,D) pooled tokens

        fused = th.cat([z_img, z_obj], dim=-1)           # (B,2D)
        return self.fuse(fused)                          # (B,out_dim)

# -------------------- Vec env --------------------
def make_vec_env(max_slots=12):
    def thunk():
        base = HackAtari("Freeway", obs_mode="ori", mode="vision", hud=False, render_mode="rgb_array")
        env = AddObjectsWrapper(base, max_slots=max_slots)
        return env
    venv = DummyVecEnv([thunk])
    venv = VecMonitor(venv)
    return venv

# -------------------- Train & quick eval --------------------
def main():
    venv = make_vec_env()
    policy_kwargs = dict(
        features_extractor_class=ObjectLLMFiLMExtractor,
        features_extractor_kwargs=dict(max_slots=12, d_model=128, out_dim=256, img_channels=64, use_deepseek=True),
        net_arch=[256, 256],
        activation_fn=nn.ReLU,
    )
    model = PPO("MultiInputPolicy", venv, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./tb_llm_film")
    model.learn(total_timesteps=200_000)

    # quick evaluation
    env = make_vec_env().envs[0]
    scores = []
    for _ in range(5):
        obs, info = env.reset()
        done = False
        R = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            R += float(r)
            done = term or trunc
        scores.append(R)
    print(f"Eval: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

if __name__ == "__main__":
    main()
