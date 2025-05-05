import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from src.models.teacher_model import TeacherModel
from src.models.LidarVAE import LidarVAE
from src.envs.envs import make_env
from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
from src.utils.helppers import convert_action


@hydra.main(config_path="config", config_name="eval", version_base="1.2")
def evaluate(cfg: DictConfig):
    # 設定表示
    print("\n=== CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    # --- 環境初期化 ---
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

    # --- モデル読み込み ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # VAE encoder
    vae = LidarVAE(lidar_dim=cfg.num_beams, latent_dim=cfg.vae.latent_dim).to(device)
    vae_ckpt = torch.load(cfg.vae.ckpt, map_location=device)
    vae.load_state_dict(vae_ckpt)
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    # TeacherModel (waypoint_dim は flattened 次元数: num_waypoints * per-point_dim)
    model = TeacherModel(
        vae_encoder=vae,
        latent_dim=cfg.vae.latent_dim,
        waypoint_dim=cfg.model.waypoint_dim,  # 例: 10点×3次元=30
        action_dim=cfg.model.action_dim,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers
    ).to(device)
    ckpt = torch.load(cfg.model.path, map_location=device)
    # Extract model_state_dict if present
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # --- 評価ループ ---
    rewards = []
    for ep in range(cfg.num_episodes):
        obs, info = env.reset()
        current_pos = info.get('current_pos', np.zeros(2, dtype=np.float32))
        prev_act = np.zeros(2, dtype=np.float32)
        total_reward = 0.0

        for step in range(cfg.num_steps):
            # LiDAR: 正規化
            scan = obs['scans'][0].astype(np.float32) / cfg.max_scan_range
            scan_tensor = torch.from_numpy(scan).unsqueeze(0).unsqueeze(0).to(device)

            # ウェイポイント取得
            wpts = map_manager.get_future_waypoints(current_pos, num_points=cfg.num_waypoints)
            if wpts.shape[0] < cfg.num_waypoints:
                pad = np.repeat(wpts[-1][None, :], cfg.num_waypoints - wpts.shape[0], axis=0)
                wpts = np.vstack([wpts, pad])
            # flatten into (1,1, waypoint_dim)
            wpts_flat = wpts.reshape(1, 1, -1).astype(np.float32)
            way_tensor = torch.from_numpy(wpts_flat).to(device)

            # prev_act 正規化
            steer_n = prev_act[0] / cfg.steer_range
            speed_n = 2.0 * prev_act[1] / cfg.speed_range - 1.0
            prev_norm = np.array([steer_n, speed_n], dtype=np.float32)
            prev_tensor = torch.from_numpy(prev_norm).unsqueeze(0).unsqueeze(0).to(device)

            # 推論
            with torch.no_grad():
                pred_norm = model(scan_tensor, way_tensor, prev_tensor)
            pred = pred_norm.squeeze().cpu().numpy()
            action = convert_action(pred, cfg.steer_range, cfg.speed_range)

            action_np = np.array(action, dtype=np.float32).reshape(1, 2)
            obs, reward, terminated, truncated, info = env.step(action_np)
            if cfg.render:
                env.render()
            total_reward += reward
            if terminated or truncated:
                break
            current_pos = info.get('current_pos', current_pos)
            prev_act = action

        print(f"Episode {ep}: Reward={total_reward:.2f}, Steps={step+1}")
        rewards.append(total_reward)

    print(f"Average Reward over {cfg.num_episodes}: {np.mean(rewards):.2f}")

if __name__ == "__main__":
    evaluate()
