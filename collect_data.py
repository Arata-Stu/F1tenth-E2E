import os
import numpy as np
import h5py
import hdf5plugin
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_env
from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
from src.planner.purePursuit import PurePursuitPlanner


@hydra.main(config_path="config", config_name="collect_data", version_base="1.2")
def main(cfg: DictConfig):
    # 設定表示
    print(OmegaConf.to_yaml(cfg))

    # --- 環境とプランナーの初期化 ---
    map_cfg = cfg.envs.map
    map_manager = MapManager(
        map_name=MAP_DICT[0],
        map_ext=map_cfg.ext,
        speed=map_cfg.speed,
        downsample=map_cfg.downsample,
        use_dynamic_speed=map_cfg.use_dynamic_speed,
        a_lat_max=map_cfg.a_lat_max,
        smooth_sigma=map_cfg.smooth_sigma
    )
    env = make_env(env_cfg=cfg.envs, map_manager=map_manager, param=cfg.vehicle)

    wheelbase = getattr(cfg.planner, 'wheelbase', 0.17145 + 0.15875)
    lookahead = getattr(cfg.planner, 'lookahead', 0.3)
    planner = PurePursuitPlanner(
        wheelbase=wheelbase,
        map_manager=map_manager,
        lookahead=lookahead,
        max_reacquire=getattr(cfg.planner, 'max_reacquire', 20.0)
    )

    # レンダリング設定
    render_flag = cfg.render
    render_mode = cfg.render_mode

    # ウェイポイント可視化設定 (config に visualize_waypoints: true を追加してください)
    if cfg.get('visualize_waypoints', False):
        plt.ion()
        fig_map = plt.figure(figsize=(6,6))
        ax_map = fig_map.add_subplot(111)
        ax_map.set_aspect('equal')
        ax_map.set_xlabel('X [m]')
        ax_map.set_ylabel('Y [m]')
        num_future = cfg.get('num_future_waypoints', 10)

    # ベース出力ディレクトリ
    base_out_dir = cfg.output_dir
    os.makedirs(base_out_dir, exist_ok=True)

    num_episodes = cfg.num_episodes
    num_steps = cfg.num_steps
    for ep in range(num_episodes):
        # マップ更新
        map_id = ep % len(MAP_DICT)
        name   = MAP_DICT[map_id]
        env.update_map(map_name=name, map_ext=map_cfg.ext)

        # マップごとのディレクトリ作成
        map_out_dir = os.path.join(base_out_dir, name)
        os.makedirs(map_out_dir, exist_ok=True)

        # 初期観測と前回アクション
        obs, _ = env.reset()
        prev_action = np.zeros((1, 2), dtype='float32')
        idx = 0
        truncated = False

        # ファイル名にエピソード番号を付与
        filename = f"{name}_speed{map_cfg.speed}_look{lookahead}_ep{ep}.h5"
        out_path = os.path.join(map_out_dir, filename)

        # HDF5ファイル＆データセット作成
        f = h5py.File(out_path, 'w')
        scans_dset = f.create_dataset(
            'scans',
            shape=(0,1080), maxshape=(None,1080),
            dtype='float32', chunks=(1,1080),
            **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )
        prev_dset = f.create_dataset(
            'prev_actions',
            shape=(0,2), maxshape=(None,2),
            dtype='float32', chunks=(1,2),
            **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )
        actions_dset = f.create_dataset(
            'actions',
            shape=(0,2), maxshape=(None,2),
            dtype='float32', chunks=(1,2),
            **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )

        # データ収集ループ
        for step in range(num_steps):
            steer, speed = planner.plan(obs, gain=0.20)
            action = np.array([steer, speed], dtype='float32').reshape(1,-1)
            scan   = obs['scans'][0].astype('float32').reshape(1,-1)

            # データ追加
            scans_dset.resize(idx+1, axis=0)
            prev_dset.resize(idx+1, axis=0)
            actions_dset.resize(idx+1, axis=0)
            scans_dset[idx]   = scan
            prev_dset[idx]    = prev_action
            actions_dset[idx] = action

            next_obs, reward, terminated, truncated, info = env.step(action)

            # ウェイポイント可視化
            if cfg.get('visualize_waypoints', False):
                current_pos = info.get('waypoint', None)
                if current_pos is not None:
                    future = map_manager.get_future_waypoints(current_pos, num_points=num_future)
                    xs, ys = future[:,0], future[:,1]
                    ax_map.clear()
                    ax_map.plot(xs, ys, marker='o', linestyle='-')
                    ax_map.scatter(current_pos[0], current_pos[1], marker='x')
                    plt.pause(0.001)

            if truncated:
                print(f"Episode {ep} truncated (failure), discarding data: {out_path}")
            if terminated or truncated:
                f.close()
                if truncated:
                    os.remove(out_path)
                break

            # 次ステップ準備
            obs = next_obs
            prev_action = action
            idx += 1
            if render_flag:
                env.render(mode=render_mode) if render_mode else env.render()

        # エピソード完了ログ
        if not truncated:
            print(f"Episode {ep} completed successfully, saved to {out_path}")


if __name__ == '__main__':
    main()