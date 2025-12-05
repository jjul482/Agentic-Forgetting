import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time
import gymnasium as gym
from hackatari.core import HackAtari
from eval import evaluate
import copy

from stable_baselines3.common.env_checker import check_env

class Float32Wrapper(gym.ObservationWrapper):
    """
    Casts HackAtari int observations to float32.
    """
    def __init__(self, env):
        super().__init__(env)
        low, high, shape = env.observation_space.low, env.observation_space.high, env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def observation(self, obs):
        return obs.astype(np.float32)

def make_env(game_name, modifications=[], rewardfunc_path=None):
    """
    Create a HackAtari environment and optionally apply a modification.
    """
    # rewardfunc_path = "freeway_custom_reward.py"
    env = HackAtari(game_name, modifs=modifications, rewardfunc_path=rewardfunc_path, obs_mode="ori")
    # Apply modification if provided
    if len(modifications) > 0:
        print(f"Applied modifications: {modifications}")
    
    # Custom reward?
    #print(env.org_reward)
    print("Obs space:", env.observation_space)
    # obs, _ = env.reset()
    #print("Obs sample:", obs[:10])
    env = Float32Wrapper(env)
    return env

def make_pixel_env(game_name, mods, mode='ori'):  # mode in {"ori","dqn"}
    # Pixels instead of object features:
    env = HackAtari(game_name, modifs=mods, obs_mode=mode, render_mode=None)  # pixels come via obs_mode
    venv = DummyVecEnv([lambda: env])
    venv = VecMonitor(venv)

    print("Obs space:", env.observation_space)

    # SB3’s CNN expects channel-first (C,H,W).
    # If the env is HWC (typical for "ori"), transpose it.
    if len(venv.observation_space.shape) == 3 and venv.observation_space.shape[-1] in (1, 3, 4):
        venv = VecTransposeImage(venv)  # HWC -> CHW

    return venv

class SimpleLoggerCallback(BaseCallback):
    def __init__(self, log_every_steps=1000, verbose=0):
        super().__init__(verbose)
        self.log_every_steps = log_every_steps
        self._last_logged = 0

    def _on_step(self):
        if (self.num_timesteps - self._last_logged) >= self.log_every_steps:
            self._last_logged = self.num_timesteps
            # Print mean reward for last 100 episodes
            ep_info = self.locals.get('infos', [])
            ep_rewards = [info['episode']['r'] for info in ep_info if 'episode' in info]
            if ep_rewards:
                print(f"[{time.strftime('%H:%M:%S')}] timesteps={self.num_timesteps}, mean_reward(last100)={np.mean(ep_rewards):.2f}")
        return True

def train_on_task(model, total_timesteps=100_000, model_save_path=None, render_mode=None):
    """
    Train PPO on a single HackAtari variation, return trained model.
    """
    cb = SimpleLoggerCallback(log_every_steps=100000)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    if model_save_path:
        model.save(model_save_path)
    return model

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    game = args["game"]
    taskA_mods = args["taskA_mods"]                # baseline task
    taskB_mods = args["taskB_mods"]  
    timesteps_per_task = args["total_timesteps"]
    model_type = args["model_type"][0]

    taskA_save_path = f"{model_type}_{game}_taskA"
    taskB_save_path = f"{model_type}_{game}_taskB"

    env_tmp = HackAtari(game)
    print(f"Available modifications for {game}:", env_tmp.available_modifications)

    results = {
        "game": game,
        "taskA_mod": taskA_mods,
        "taskB_mod": taskB_mods,
        "timesteps_per_task": timesteps_per_task,
    }

    venvA = make_pixel_env(game, taskA_mods)
    venvB = make_pixel_env(game, taskB_mods)
    modelA, modelB = None, None
    if model_type == "ppo":
        modelA = PPO('MlpPolicy', venvA, verbose=1, tensorboard_log="./tb_logs", device="cuda")
        modelB = PPO('MlpPolicy', venvB, verbose=1, tensorboard_log="./tb_logs", device="cuda")

    print("Training on Task A (baseline)...")
    modelA = train_on_task(modelA, total_timesteps=timesteps_per_task, model_save_path=taskA_save_path)
    print("Evaluating agent trained on Task A:")
    meanA, stdA = evaluate(venvA, modelA, n_episodes=20)
    print(f"Task A eval (after A train): mean={meanA:.2f} ±{stdA:.2f}")
    results['A_after_A_mean'] = meanA

    # Evaluate A agent on Task B (transfer baseline)
    envB_for_eval = make_env(game, taskB_mods)
    meanA_on_B, _ = evaluate(envB_for_eval, modelA, n_episodes=20)
    print(f"Task A agent on Task B (zero-shot): mean={meanA_on_B:.2f}")
    results['A_on_B_before_B_train'] = meanA_on_B

    # Train new model on Task B
    print("Training on Task B (modified)...")
    modelB = train_on_task(modelB, total_timesteps=timesteps_per_task, model_save_path=taskB_save_path)
    meanB_afterB, _ = evaluate(venvB, modelB, n_episodes=20)
    print(f"Task B eval (after B train): mean={meanB_afterB:.2f}")
    results['B_after_B_mean'] = meanB_afterB

    envA_for_eval = make_env(game, taskA_mods)
    meanB_on_A_afterB, _ = evaluate(envA_for_eval, modelB, n_episodes=20)
    print(f"Model after B-training evaluated on A: mean={meanB_on_A_afterB:.2f}")
    results['A_after_B_mean'] = meanB_on_A_afterB
    #watch_agent(envA, modelA, n_episodes=1)

    # Compute forgetting metric (simple)
    results['forgetting_A'] = results['A_after_A_mean'] - results['A_after_B_mean']

    # Save results
    out_path = f"{model_type}_{game}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to", out_path)
    print(json.dumps(results, indent=2))