import numpy as np

def reward_function(env, info=None):
    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, '_get_obs') else None
    penalty = 0.0
    # Penalize collision if info is provided and collision detected
    if info is not None and ("collision" in info or "hit" in info):
        if info.get("collision", False) or info.get("hit", False):
            penalty = -1.0  # Set your penalty value here

    if obs is not None:
        agent_color = np.array([252, 252, 84])
        mask = np.all(obs == agent_color, axis=-1)
        agent_rows = np.where(mask)[0]
        if len(agent_rows) > 0:
            agent_y = float(np.mean(agent_rows))
            road_top = 0
            road_bottom = 159
            norm_dist = 1.0 - (agent_y - road_top) / (road_bottom - road_top)
            return float(norm_dist) + penalty
    return penalty