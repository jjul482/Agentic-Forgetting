import os
import math
import time
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import cv2  # pip install opencv-python

from stable_baselines3.common.vec_env import VecFrameStack
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecVideoRecorder, VecNormalize
from stable_baselines3.common.utils import set_random_seed


# -----------------------------
# Intrinsic exploration wrapper
# -----------------------------
class HashBonus(gym.Wrapper):
    """
    Simple count-based exploration for pixels:
    - Downsample + binarize the stacked grayscale frames
    - SimHash to a 64-bit key
    - Reward bonus = beta / sqrt(N(s))
    """
    def __init__(self, env: gym.Env, beta: float = 0.01,
                 ds_size: Tuple[int, int] = (21, 8)):
        super().__init__(env)
        self.beta = beta
        self.ds_w, self.ds_h = ds_size
        self.counts: Dict[int, int] = {}

    @staticmethod
    def _preprocess(obs: np.ndarray, ds_w: int, ds_h: int) -> np.ndarray:
        """
        obs: (H, W, C_stack) grayscale uint8 (from AtariPreprocessing + FrameStack)
        We max-pool across the stack then downsample.
        """
        # Max over time-dimension channel stack to emphasize moving objects
        pooled = obs.max(axis=2).astype(np.uint8)  # (H, W)
        # Downsample to very small grid
        tiny = cv2.resize(pooled, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        # Binarize
        _, tiny_bin = cv2.threshold(tiny, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return tiny_bin.astype(np.uint8)

    @staticmethod
    def _simhash(arr: np.ndarray) -> int:
        """64-bit SimHash of a small binary array."""
        # Simple FNV-1a 64-bit over bytes
        data = arr.tobytes()
        h = 1469598103934665603  # FNV offset basis
        for b in data:
            h ^= b
            h = (h * 1099511628211) & ((1 << 64) - 1)
        return h

    def step(self, action):
        obs, extr, terminated, truncated, info = self.env.step(action)

        # expect obs shape (H, W, stack) as uint8
        tiny = self._preprocess(obs, self.ds_w, self.ds_h)
        key = self._simhash(tiny)

        cnt = self.counts.get(key, 0) + 1
        self.counts[key] = cnt

        bonus = self.beta / math.sqrt(cnt)
        rew = extr + bonus

        # You can store the bonus for logging
        info = dict(info)  # copy
        info["intrinsic_bonus"] = bonus
        return obs, rew, terminated, truncated, info


# -----------------------------
# Reward shaping (optional)
# -----------------------------
class GentleCollisionPenalty(gym.Wrapper):
    """
    Adds a small per-step penalty after a collision to discourage “mindless UP”.
    Detecting collisions directly from ALE pixels is non-trivial; instead,
    we approximate by penalizing immediately after a reward decrease (rare in Freeway),
    or after a 'stuck' window where no vertical progress is observed.

    This is conservative and won’t change the optimal policy.
    """
    def __init__(self, env: gym.Env, penalty: float = -0.005, window:int = 20):
        super().__init__(env)
        self.penalty = penalty
        self.window = window
        self.last_score = 0.0
        self.progress_buffer = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = 0.0
        self.progress_buffer.clear()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track score deltas (Freeway yields +1 upon crossing)
        score = info.get("episode_score", 0.0) if "episode_score" in info else 0.0
        # Not all ALE wrappers populate this; fallback to env-wide episodic return
        # If not provided, we skip score logic and only use "stuck" heuristic.

        # Heuristic “stuck” detector: if many consecutive zero rewards,
        # gently penalize to break always-UP collapse.
        r_is_zeroish = (abs(reward) < 1e-6)
        self.progress_buffer.append(0 if r_is_zeroish else 1)
        if len(self.progress_buffer) > self.window:
            self.progress_buffer.pop(0)

        stuck = (sum(self.progress_buffer) == 0 and len(self.progress_buffer) == self.window)
        shaped = reward + (self.penalty if stuck else 0.0)

        if "shaping_stuck" in info:
            # preserve keys if env already uses them
            info = dict(info)
            info["shaping_stuck"] = info["shaping_stuck"] or stuck
        else:
            info = dict(info)
            info["shaping_stuck"] = stuck

        return obs, shaped, terminated, truncated, info


# -----------------------------
# Env factory
# -----------------------------
def make_freeway_env(seed: int, rank: int, frame_stack: int = 4,
                     sticky_prob: float = 0.25,
                     add_hash_bonus: bool = True,
                     add_penalty: bool = True) -> Callable[[], gym.Env]:
    """
    Returns a thunk that creates one Atari Freeway env with:
    - No frame-skip inside ALE (we use MaxAndSkip in AtariPreprocessing)
    - DeepMind preprocessing: grayscale, scale to 84x84, frame-skip=4
    - FrameStack (4)
    - Optional intrinsic bonus + gentle penalty
    """
    def _thunk():
        env = gym.make(
            "ALE/Freeway-v5",
            frameskip=1,  # handled by AtariPreprocessing’s frame_skip
            repeat_action_probability=sticky_prob,
            full_action_space=False,  # standard 18-Actions not needed
            render_mode=None
        )
        # DeepMind-style preprocessing
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            grayscale_obs=True,
            terminal_on_life_loss=False,
            scale_obs=False  # keep uint8; SB3 CNN expects uint8
        )
        # Obs shape now (84, 84); stack to (84, 84, 4)
        env = VecFrameStack(env, num_stack=frame_stack)

        # Optional shaping/exploration
        if add_penalty:
            env = GentleCollisionPenalty(env, penalty=-0.005, window=24)
        if add_hash_bonus:
            env = HashBonus(env, beta=0.01, ds_size=(21, 8))

        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1000)

        # Seeding
        env.reset(seed=seed + rank)
        return env
    return _thunk


# -----------------------------
# Training config
# -----------------------------
@dataclass
class TrainConfig:
    total_timesteps: int = 10_000_000
    n_envs: int = 8
    seed: int = 0
    logdir: str = "./runs/freeway_ppo"
    ckpt_every_steps: int = 250_000
    eval_every_steps: int = 250_000
    video_every_steps: int = 1_000_000
    video_length: int = 5000  # frames
    gamma: float = 0.999
    gae_lambda: float = 0.95
    n_steps: int = 8192
    batch_size: int = 512
    learning_rate: float = 1e-4
    clip_range: float = 0.1
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def main():
    cfg = TrainConfig()
    os.makedirs(cfg.logdir, exist_ok=True)
    set_random_seed(cfg.seed)

    # -----------------------
    # Vectorized train envs
    # -----------------------
    env_fns = [make_freeway_env(seed=cfg.seed, rank=i) for i in range(cfg.n_envs)]
    venv = SubprocVecEnv(env_fns) if cfg.n_envs > 1 else DummyVecEnv(env_fns)
    venv = VecMonitor(venv)
    # Normalize returns (not observations, since we feed uint8 CNN)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # -----------------------
    # Eval env (separate RNG)
    # -----------------------
    eval_env_fn = make_freeway_env(seed=cfg.seed + 10_000, rank=0,
                                   add_hash_bonus=False,  # evaluate on extrinsic only
                                   add_penalty=False)
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    # VERY IMPORTANT: sync stats from train vecnorm to eval vecnorm
    eval_env.obs_rms = venv.obs_rms
    eval_env.ret_rms = venv.ret_rms

    # Optional: record videos every N steps (evaluation-only to avoid training slowdown)
    video_dir = os.path.join(cfg.logdir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    eval_env_video = VecVideoRecorder(
        eval_env,
        video_folder=video_dir,
        record_video_trigger=lambda step: step > 0 and step % cfg.video_every_steps == 0,
        video_length=cfg.video_length
    )

    # -----------------------
    # PPO agent
    # -----------------------
    policy_kwargs = dict(
        # SB3 default CnnPolicy is fine; you can tweak net_arch if desired.
        # net_arch=[dict(pi=[512, 256], vf=[512, 256])]
    )

    model = PPO(
        "CnnPolicy",
        venv,
        verbose=1,
        seed=cfg.seed,
        tensorboard_log=cfg.logdir,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs
    )

    # -----------------------
    # Callbacks
    # -----------------------
    ckpt_cb = CheckpointCallback(
        save_freq=cfg.ckpt_every_steps // cfg.n_envs,  # per-env step accounting
        save_path=os.path.join(cfg.logdir, "checkpoints"),
        name_prefix="freeway_ppo"
    )
    eval_cb = EvalCallback(
        eval_env_video,
        best_model_save_path=os.path.join(cfg.logdir, "best"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=cfg.eval_every_steps // cfg.n_envs,
        deterministic=False,
        render=False,
        n_eval_episodes=10
    )

    # -----------------------
    # Train
    # -----------------------
    print("Starting training...")
    start = time.time()
    model.learn(total_timesteps=cfg.total_timesteps, callback=[ckpt_cb, eval_cb], progress_bar=True)
    elapsed = (time.time() - start) / 3600
    print(f"Done. Elapsed ~{elapsed:.2f} hours")

    # Save final policy
    model.save(os.path.join(cfg.logdir, "final_policy"))

    # Close envs
    venv.close()
    eval_env_video.close()


if __name__ == "__main__":
    """
    Quick environment setup (conda):

        conda create -n freeway python=3.10 -y
        conda activate freeway
        pip install "gymnasium[atari,accept-rom-license]" stable-baselines3 torch opencv-python

    Optional (TensorBoard):
        pip install tensorboard
        tensorboard --logdir ./runs/freeway_ppo

    Then:
        python train_freeway_ppo.py
    """
    main()
