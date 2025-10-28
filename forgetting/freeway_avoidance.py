# freeway_llm_guided_policy.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from hackatari.core import HackAtari
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

# ---------- 1) Env wrapper: add object slots to observation ----------
class AddObjectsWrapper(gym.ObservationWrapper):
    """
    Returns Dict obs: {"pixels": HxWxC uint8, "objects": (N, D) float32}
    - N slots, padded with zeros; D features per object (x,y,w,h,type_id).
    """
    def __init__(self, env, max_slots=12):
        super().__init__(env)
        self.max_slots = max_slots
        self.obj_dim = 5  # [x, y, w, h, type_id]
        # pixels space (H,W,C) remains identical
        pix_space = env.observation_space  # Box(0,255,(210,160,3),uint8)
        obj_space = spaces.Box(low=0.0, high=255.0, shape=(self.max_slots, self.obj_dim), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "pixels": pix_space,
            "objects": obj_space
        })

    def _extract_objects(self):
        slots = []
        for o in getattr(self.env, "objects", []):
            # Robust attribute access
            x = float(getattr(o, "x", 0)); y = float(getattr(o, "y", 0))
            w = float(getattr(o, "w", 0)); h = float(getattr(o, "h", 0))
            name = getattr(o, "category", type(o).__name__).lower()
            # crude type ids (replace with a consistent mapping or embed names via LLM)
            if "player" in name or "chicken" in name: tid = 1
            elif "car" in name or "vehicle" in name or "truck" in name: tid = 2
            elif "goal" in name or "flag" in name or "finish" in name: tid = 3
            else: tid = 0
            if w > 0 and h > 0:
                slots.append([x, y, w, h, tid])
        # pad / trim to max_slots
        if len(slots) < self.max_slots:
            slots += [[0,0,0,0,0]] * (self.max_slots - len(slots))
        else:
            slots = slots[:self.max_slots]
        return np.array(slots, dtype=np.float32)

    def observation(self, obs):
        # obs is pixels (HWC)
        objs = self._extract_objects()
        return {"pixels": obs, "objects": objs}

# ---------- 2) LLM guidance (stub) ----------
class GuidanceLLM(nn.Module):
    """
    Frozen guidance that returns:
      - alpha: (N,) per-object salience
      - B: (N,N) attention bias matrix added to attention scores
    Replace forward() with a real LLM call that reads object names/coords.
    """
    def __init__(self, max_slots=12):
        super().__init__()
        self.max_slots = max_slots

    @th.no_grad()
    def forward(self, obj_slots: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        obj_slots: (B, N, 5) with [x,y,w,h,tid]
        Returns:
          alpha: (B, N) in [-1, +1] (salience; used to scale keys/values)
          B:     (B, N, N) additive bias to attention scores
        """
        Bsz, N, D = obj_slots.shape
        tid = obj_slots[..., 4]  # (B,N)
        # salience: cars high (+), others low; normalize to [-1,1]
        is_car = (tid == 2).float()
        is_goal = (tid == 3).float()
        is_player = (tid == 1).float()
        alpha = 0.6 * is_car + 0.3 * is_goal + 0.1 * is_player  # [0..1]
        alpha = 2.0 * alpha - 1.0  # [-1..1]

        # bias: discourage attention between player and cars (avoid); encourage player->goal
        B = th.zeros(Bsz, N, N, device=obj_slots.device)
        # find indices per batch
        for b in range(Bsz):
            player_idx = (th.nonzero(is_player[b], as_tuple=False).flatten())
            car_idx = (th.nonzero(is_car[b], as_tuple=False).flatten())
            goal_idx = (th.nonzero(is_goal[b], as_tuple=False).flatten())
            for p in player_idx.tolist():
                for c in car_idx.tolist():
                    B[b, p, c] -= 0.8  # discourage attending to cars (danger)
                for g in goal_idx.tolist():
                    B[b, p, g] += 0.8  # encourage attending to goal
        return alpha, B

# ---------- 3) Attention with bias ----------
class BiasedSelfAttention(nn.Module):
    """
    Self-attention over object tokens with additive bias and salience scaling.
    """
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
        """
        x: (B,N,D)
        alpha: (B,N) in [-1,1] → rescale keys/values; optional
        bias: (B,N,N) added to attention logits; optional
        """
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.n_heads, self.d_k).transpose(1,2)  # (B,H,N,d)
        k = self.k(x).view(B, N, self.n_heads, self.d_k).transpose(1,2)
        v = self.v(x).view(B, N, self.n_heads, self.d_k).transpose(1,2)

        if alpha is not None:
            # broadcast (B,1,N,1)
            scale = (1.0 + alpha.unsqueeze(1).unsqueeze(-1))  # bias keys/values by salience
            k = k * scale
            v = v * scale

        attn_logits = (q @ k.transpose(-2, -1)) / np.sqrt(self.d_k)  # (B,H,N,N)
        if bias is not None:
            attn_logits = attn_logits + bias.unsqueeze(1)  # share across heads
        attn = attn_logits.softmax(dim=-1)
        y = attn @ v  # (B,H,N,d)
        y = y.transpose(1,2).contiguous().view(B, N, D)
        return self.o(y)

# ---------- 4) SB3 features extractor ----------
class ObjectLLMAttnExtractor(BaseFeaturesExtractor):
    """
    - CNN over pixels -> z_img
    - project objects -> z_obj
    - LLM guidance -> (alpha, B)
    - Biased self-attention over z_obj -> z_obj*
    - fuse: [pool(z_img), pool(z_obj*)] -> final features
    """
    def __init__(self, obs_space: spaces.Dict, max_slots=12, d_model=128, out_dim=256):
        super().__init__(obs_space, features_dim=out_dim)
        self.max_slots = max_slots
        # simple CNN (210x160x3 -> 20x15x64 -> pooled)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.img_pool = nn.AdaptiveAvgPool2d((1,1))
        self.img_proj = nn.Linear(64, d_model)

        # object encoder
        self.obj_proj = nn.Linear(5, d_model)
        self.obj_attn = BiasedSelfAttention(d_model=d_model, n_heads=4)
        self.llm = GuidanceLLM(max_slots=max_slots)  # frozen prior

        # fusion
        self.fuse = nn.Sequential(
            nn.LayerNorm(2*d_model),
            nn.Linear(2*d_model, out_dim), nn.ReLU()
        )

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        # pixels: (B,H,W,C or B,C,H,W depending on VecTransposeImage)
        px = obs["pixels"]
        # ensure channel-first
        if px.shape[1] != 3:
            # HWC -> CHW
            px = px.permute(0, 3, 1, 2).contiguous()
        px = px.float() / 255.0

        # objects: (B,N,5)
        objs = obs["objects"].float()  # already 0..255 range

        # image path
        z = self.cnn(px)                        # (B,64,h,w)
        z = self.img_pool(z).squeeze(-1).squeeze(-1)  # (B,64)
        z_img = self.img_proj(z)                # (B,D)

        # object path + guidance
        z_obj = self.obj_proj(objs)             # (B,N,D)
        with th.no_grad():
            alpha, bias = self.llm(objs)        # (B,N), (B,N,N)
        z_obj = self.obj_attn(z_obj, alpha=alpha, bias=bias)  # (B,N,D)
        z_obj = z_obj.mean(dim=1)               # simple mean-pool over tokens -> (B,D)

        fused = th.cat([z_img, z_obj], dim=-1)  # (B,2D)
        return self.fuse(fused)                 # (B,out_dim)

# ---------- 5) Make vectorized env ----------
def make_vec_env(max_slots=12):
    def thunk():
        base = HackAtari("Freeway", obs_mode="ori", mode="vision", hud=False, render_mode="rgb_array")
        env = AddObjectsWrapper(base, max_slots=max_slots)
        return env
    venv = DummyVecEnv([thunk])
    venv = VecMonitor(venv)
    # SB3 expects CHW for pixels branch inside extractor? We convert inside extractor, so no VecTransposeImage needed.
    return venv

# ---------- 6) Train PPO with MultiInputPolicy ----------
def run():
    venv = make_vec_env()
    policy_kwargs = dict(
        features_extractor_class=ObjectLLMAttnExtractor,
        features_extractor_kwargs=dict(max_slots=12, d_model=128, out_dim=256),
        net_arch=[256, 256],  # policy+value heads MLP sizes
        activation_fn=nn.ReLU
    )
    model = PPO("MultiInputPolicy", venv, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./tb_llm_attn")
    model.learn(total_timesteps=200_000)

    # quick eval
    env = make_vec_env().envs[0]
    rets = []
    for _ in range(5):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += float(r)
            done = term or trunc
        rets.append(ep_ret)
    print(f"Eval: {np.mean(rets):.2f} ± {np.std(rets):.2f}")

if __name__ == "__main__":
    run()
