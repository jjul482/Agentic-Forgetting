from hackatari.core import HackAtari
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import json
import argparse

def watch_agent(env, model, n_episodes=1, deterministic=True):
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

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)

    model_type = args["model_type"][0]
    game = args["game"]
    model_task = args["model_task"]
    env_task = args["env_task"]

    save_path = f"{model_type}_{game}_task{model_task}"

    env = HackAtari(game,modifs=args[f"task{env_task}_mods"] , render_mode="human")
    venv = DummyVecEnv([lambda: env])
    venv = VecMonitor(venv)
    model = PPO('MlpPolicy', venv, verbose=1, tensorboard_log="./tb_logs", device="cuda")
    model = model.load(f"{save_path}.zip")
    watch_agent(env, model, n_episodes=1)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./exps/ppo_boxing_one_armed.json',)
    parser.add_argument('--model_task', type=str, default='B',)
    parser.add_argument('--env_task', type=str, default='B',)
    return parser

if __name__ == "__main__":
    main()