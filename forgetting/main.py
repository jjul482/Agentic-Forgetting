import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time
import gymnasium as gym
from hackatari.core import HackAtari

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

def make_env(game_name, modifications=[], rewardfunc_path=None, render_mode=None):
    """
    Create a HackAtari environment and optionally apply a modification.
    """
    rewardfunc_path = "freeway_custom_reward.py"
    env = HackAtari(game_name, modifs=modifications, rewardfunc_path=rewardfunc_path, render_mode=render_mode)
    # Apply modification if provided
    if len(modifications) > 0:
        print(f"Applied modifications: {modifications}")
    
    # Custom reward?
    #print(env.org_reward)
    #print("Obs space:", env.observation_space)
    obs, _ = env.reset()
    print("Obs sample:", obs[:10])
    env = Float32Wrapper(env)
    return env

def evaluate(env, model, n_episodes=20, deterministic=True):
    """
    Return mean return and std over n_episodes.
    """
    returns = []
    for ep in range(n_episodes):
        reset_out = env.reset()
        # Gymnasium: reset() -> (obs, info)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        ep_ret = 0.0
        while True:
            # SB3 expects a numpy array (no tuple)
            action, _ = model.predict(obs, deterministic=deterministic)
            step_out = env.step(action)
            # Gymnasium: step() -> (obs, reward, terminated, truncated, info)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:  # legacy Gym
                obs, reward, done, info = step_out
            ep_ret += float(np.array(reward).sum()) if isinstance(reward, (list, np.ndarray)) else float(reward)
            if done:
                break
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))

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

def watch_agent(env, model, n_episodes=1, deterministic=True):
    """
    Render the environment and watch the agent play.
    """
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            env.render()  # Show the game window
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        print(f"Episode {ep+1} return: {ep_ret}")
    env.close()

def train_on_task(game, modifications, total_timesteps=100_000, model_save_path=None, render_mode=None, use_model=None):
    """
    Train PPO on a single HackAtari variation, return trained model.
    """
    env = make_env(game, modifications, render_mode=render_mode)
    venv = DummyVecEnv([lambda: env])
    venv = VecMonitor(venv)  # gives episode returns/lengths
    if use_model:
        model = use_model
    else:
        model = PPO('MlpPolicy', venv, verbose=1, tensorboard_log="./tb_logs", device="cuda")
    cb = SimpleLoggerCallback(log_every_steps=50000)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    if model_save_path:
        model.save(model_save_path)
    return model, env  # return env for convenience (for evaluation)

def main():
    # Freeway baseline vs modification 'all_black_cars'
    game = "Freeway"
    taskA_mods = []                # baseline task
    taskB_mods = ["all_black_cars"]
    timesteps_per_task = 200_000

    env_tmp = HackAtari(game)
    print(f"Available modifications for {game}:", env_tmp.available_modifications)


    results = {
        "game": game,
        "taskA_mod": taskA_mods,
        "taskB_mod": taskB_mods,
        "timesteps_per_task": timesteps_per_task,
    }

    print("Training on Task A (baseline)...")
    modelA, envA = train_on_task(game, taskA_mods, total_timesteps=timesteps_per_task, model_save_path="ppo_taskA")
    print("Evaluating agent trained on Task A:")
    meanA, stdA = evaluate(envA, modelA, n_episodes=20)
    print(f"Task A eval (after A train): mean={meanA:.2f} ±{stdA:.2f}")
    results['A_after_A_mean'] = meanA
    # After training
    envA_vis = make_env(game, taskA_mods, render_mode="human")
    watch_agent(envA_vis, modelA, n_episodes=1)

    # Evaluate A agent on Task B (transfer baseline)
    envB_for_eval = make_env(game, taskB_mods)
    meanA_on_B, _ = evaluate(envB_for_eval, modelA, n_episodes=20)
    print(f"Task A agent on Task B (zero-shot): mean={meanA_on_B:.2f}")
    results['A_on_B_before_B_train'] = meanA_on_B

    # Now train on Task B
    print("Training on Task B (modified)...")
    modelB, envB = train_on_task(game, taskB_mods, total_timesteps=timesteps_per_task, model_save_path="ppo_taskB")
    meanB_afterB, _ = evaluate(envB, modelB, n_episodes=20)
    print(f"Task B eval (after B train): mean={meanB_afterB:.2f}")
    results['B_after_B_mean'] = meanB_afterB

    # Now evaluate the *same* model trained on B on Task A -> measure forgetting of A
    envA_for_eval = make_env(game, taskA_mods)
    meanB_on_A_afterB, _ = evaluate(envA_for_eval, modelB, n_episodes=20)
    print(f"Model after B-training evaluated on A: mean={meanB_on_A_afterB:.2f}")
    results['A_after_B_mean'] = meanB_on_A_afterB
    watch_agent(envA, modelA, n_episodes=1)

    # Compute forgetting metric (simple)
    results['forgetting_A'] = results['A_after_A_mean'] - results['A_after_B_mean']

    # Save results
    out_path = "forgetting_run_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to", out_path)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
