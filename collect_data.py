import os
import numpy as np
import h5py
import hdf5plugin
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_env
from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
from src.planner.purePursuit import PurePursuitPlanner

@hydra.main(config_path="config", config_name="collect_data", version_base="1.2")
def main(cfg: DictConfig):
    # 設定表示
    print(OmegaConf.to_yaml(cfg))

    # 実行ごとに固有のランIDディレクトリを作成
    base_out = cfg.output_dir
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(base_out, run_id)
    os.makedirs(out_root, exist_ok=True)

    # --- 環境とプランナーの初期化 ---
    map_cfg     = cfg.envs.map
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

    # PurePursuit 初期化
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

    num_episodes  = cfg.num_episodes
    num_steps     = cfg.num_steps
    # ウェイポイントの数（config に num_waypoints=None ならデフォルト10）
    num_waypoints = cfg.get('num_waypoints', 10)

    # --- マップごとの出現回数カウンタを用意 ---
    map_counters = {m: 0 for m in MAP_DICT.values()}

    for ep in range(num_episodes):
        # マップ選択・リセット
        map_id = ep % len(MAP_DICT)
        name   = MAP_DICT[map_id]
        env.update_map(map_name=name, map_ext=map_cfg.ext)
        obs, info = env.reset()

        # このマップの出現回数を取得してインクリメント
        count = map_counters[name]
        map_counters[name] += 1

        # 初期値準備
        prev_action = np.zeros((1, 2), dtype='float32')
        current_pos = info.get('current_pos', np.array([0.0, 0.0], dtype='float32'))
        idx         = 0
        truncated   = False

        # エピソード用出力ディレクトリ
        ep_dir = os.path.join(out_root, name)
        os.makedirs(ep_dir, exist_ok=True)

        # HDF5 ファイル作成（マップごとのカウントで一意化）
        filename = f"{name}_speed{map_cfg.speed}_look{lookahead}_run{count}.h5"
        out_path = os.path.join(ep_dir, filename)
        f = h5py.File(out_path, 'w')

        # データセット定義
        pos_dset = f.create_dataset(
            'positions', shape=(0, 2), maxshape=(None, 2),
            dtype='float32', chunks=(1, 2),
            **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )
        scans_dset = f.create_dataset(
            'scans', shape=(0, cfg.num_beams), maxshape=(None, cfg.num_beams),
            dtype='float32', chunks=(1, cfg.num_beams),
            **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )
        # ← 追加：waypoints データセット
        wpt_dset = f.create_dataset(
            'waypoints', shape=(0, num_waypoints, 2), maxshape=(None, num_waypoints, 2),
            dtype='float32', chunks=(1, num_waypoints, 2),
            **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )
        prev_dset = f.create_dataset(
            'prev_actions', shape=(0, 2), maxshape=(None, 2),
            dtype='float32', chunks=(1, 2),
            **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )
        actions_dset = f.create_dataset(
            'actions', shape=(0, 2), maxshape=(None, 2),
            dtype='float32', chunks=(1, 2),
            **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )

        # データ収集ループ
        for step in range(num_steps):
            # PurePursuit で行動計算
            steer, speed = planner.plan(obs, gain=cfg.planner.gain)
            action = np.array([steer, speed], dtype='float32').reshape(1, 2)
            scan   = obs['scans'][0].astype('float32').reshape(1, cfg.num_beams)

            # --- ここからウェイポイント取得＆整形 ---
            # MapManager のメソッドを使って将来の点を取る
            wpts = map_manager.get_future_waypoints(
                current_pos, num_points=num_waypoints
            ).astype('float32')  # (N,2)
            # 足りない場合は最後の点をパディング
            if wpts.shape[0] < num_waypoints:
                pad = np.repeat(wpts[-1][None, :], num_waypoints - wpts.shape[0], axis=0)
                wpts = np.vstack([wpts, pad])
            # バッチ次元追加 → (1, N, 2)
            wpts = wpts.reshape(1, num_waypoints, 2)
            # ―――――――――――――――――――――――

            # 各データをリサイズ＆保存
            for dset, data in zip(
                [pos_dset, scans_dset, wpt_dset, prev_dset, actions_dset],
                [current_pos.reshape(1, 2), scan, wpts, prev_action, action]
            ):
                dset.resize(idx + 1, axis=0)
                dset[idx] = data

            # ステップ実行
            next_obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                print(f"Episode {ep} truncated, discarding: {out_path}")
            if terminated or truncated:
                f.close()
                if truncated:
                    os.remove(out_path)
                break

            # 次ステップ準備
            obs         = next_obs
            prev_action = action
            current_pos = info.get('current_pos', current_pos)
            idx        += 1

            # レンダリング
            if render_flag:
                env.render(mode=render_mode) if render_mode else env.render()

        if not truncated:
            print(f"Episode {ep} completed, saved: {out_path}")

        f.close()

if __name__ == '__main__':
    main()
