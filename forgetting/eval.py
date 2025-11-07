import numpy as np

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