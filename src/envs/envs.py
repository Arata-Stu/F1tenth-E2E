from typing import Dict
from omegaconf import DictConfig
from f1tenth_gym.f110_env import F110Env
from .wrapper import F110Wrapper
from f1tenth_gym.maps.map_manager import MapManager

def make_env(env_cfg: DictConfig, map_manager: MapManager,  param: Dict):
    map_name = map_manager.map_path
    max_ext = map_manager.map_ext
    num_beams = env_cfg.num_beams
    num_agents = env_cfg.num_agents
    env = F110Env(map=map_name, map_ext=max_ext, num_beams=num_beams, num_agents=num_agents, params=param)
    env = F110Wrapper(env, map_manager=map_manager)

    return env