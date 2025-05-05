import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from f1tenth_gym.maps.map_manager import MapManager
from f1tenth_gym.f110_env import F110Env
from src.envs.wrapper import F110Wrapper
from src.planner.purePursuit import PurePursuitPlanner
from f1tenth_gym.maps.map_manager import MAP_DICT

@hydra.main(config_path="config", config_name="lane_follow", version_base="1.2")
def main(cfg: DictConfig):
    # --- 設定表示 ---
    print(OmegaConf.to_yaml(cfg))

    # --- Matplotlib センサ可視化設定 ---
    if cfg.visualize_lidar:
        plt.ion()
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)
        line, = ax.plot([], [], lw=2)
        ax.set_ylim(0, cfg.max_lidar_range)
    # --- MapManager & 環境初期化 ---
    map_cfg = cfg.envs.map
    

    map_manager = MapManager(
        map_name=MAP_DICT[0],
        map_ext=map_cfg.ext,
        line_type=map_cfg.line_type,
        speed=map_cfg.speed,
        downsample=map_cfg.downsample,
        use_dynamic_speed=map_cfg.use_dynamic_speed,
        a_lat_max=map_cfg.a_lat_max,
        smooth_sigma=map_cfg.smooth_sigma
    )
    env = F110Env(
        map=map_manager.map_path,
        map_ext=map_manager.map_ext,
        num_beams=cfg.num_beams,
        num_agents=1,
        params=cfg.vehicle
    )
    env = F110Wrapper(env, map_manager=map_manager)

    # --- Planner 初期化 ---
    planner = PurePursuitPlanner(
        wheelbase=cfg.planner.wheelbase,
        map_manager=map_manager,
        lookahead=cfg.planner.lookahead,
        max_reacquire=cfg.planner.max_reacquire
    )

    # --- マップ切り替えループ ---
    for map_name in MAP_DICT.values():
        print(f"\n=== マップ: {map_name} ===")

        # マップ更新＆リセット
        env.update_map(map_name=map_name, map_ext=map_cfg.ext)
        obs, info = env.reset()
        terminated = truncated = False

        # メインループ：センサ可視化 + 環境描画
        while not (terminated or truncated):
            # アクション計算
            steer, speed_cmd = planner.plan(obs, gain=cfg.planner.gain)
            action = np.array([[steer, speed_cmd]], dtype="float32")

            # ステップ実行
            next_obs, reward, terminated, truncated, info = env.step(action)

            if cfg.visualize_lidar:
                scan   = obs["scans"][0]
                angles = np.linspace(-cfg.lidar_fov, cfg.lidar_fov, cfg.num_beams)
                line.set_data(angles, scan)
                plt.pause(0.0001)

            # Gym 環境可視化
            env.render(mode=cfg.render_mode) if cfg.render_mode else env.render()

            obs = next_obs

        print(f"マップ『{map_name}』終了  terminated={terminated}, truncated={truncated}")
        

if __name__ == "__main__":
    main()
