import os
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from hydra.utils import to_absolute_path

from src.models.teacher_model import TeacherModel
from src.models.LidarVAE import LidarVAE
from src.data.h5_dataset import H5SequenceDataset

@hydra.main(config_path="config", config_name="train_teacher", version_base="1.2")
def train(cfg: DictConfig):
    # 設定表示
    print("\n=== CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard writer
    log_dir = cfg.train.get('log_dir', 'runs')
    writer = SummaryWriter(log_dir=log_dir)

    # --- Dataset / DataLoader ---
    dataset = H5SequenceDataset(root_dir=cfg.data.root_dir, seq_len=cfg.data.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )

    # --- LidarVAE 読み込み & 凍結 ---
    vae = LidarVAE(lidar_dim=cfg.data.num_beams, latent_dim=cfg.vae.latent_dim).to(device)
    vae_ckpt = torch.load(to_absolute_path(cfg.vae.ckpt), map_location=device)
    vae.load_state_dict(vae_ckpt)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # --- TeacherModel 初期化 ---
    model = TeacherModel(
        vae_encoder=vae,
        latent_dim=cfg.vae.latent_dim,
        waypoint_dim=cfg.model.waypoint_dim,
        action_dim=cfg.model.action_dim,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
    ).to(device)

    # --- Optimizer & Loss ---
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr)
    criterion = nn.SmoothL1Loss()

    # normalization params
    max_range = cfg.data.get('max_scan_range', 30.0)
    steer_range = cfg.data.get('steer_range', 0.4)
    speed_range = cfg.data.get('speed_range', 10.0)

    # --- 学習ループ ---
    global_step = 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.train.epochs}", leave=False)
        for scans, waypts, prev_act, true_act, positions in loop:
            # デバイスと正規化
            scans = scans.to(device) / max_range       # (B, T, num_beams)
            prev_act = prev_act.to(device)
            true_act = true_act.to(device)
            positions = positions.to(device)           # (B, T, 2)

            # true_act を [-1,1] にマッピング
            steer_norm = true_act[...,0] / steer_range
            speed_norm = 2.0 * true_act[...,1] / speed_range - 1.0
            label = torch.stack([steer_norm, speed_norm], dim=-1)

            # prev_act 同様に正規化
            p_steer = prev_act[...,0] / steer_range
            p_speed = 2.0 * prev_act[...,1] / speed_range - 1.0
            prev_norm = torch.stack([p_steer, p_speed], dim=-1)

            # waypts を相対座標に変換
            B, T, wp_dim = waypts.shape
            N = wp_dim // 3
            w = waypts.view(B, T, N, 3).to(device)
            rel_xy = w[..., :2] - positions.unsqueeze(2)
            yaw    = w[..., 2:3]
            w_rel = torch.cat([rel_xy, yaw], dim=-1)
            waypts_flat = w_rel.view(B, T, N*3)

            # forward
            pred = model(scans, waypts_flat, prev_norm)
            loss = criterion(pred, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * scans.size(0)
            global_step += 1

            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch:02d}/{cfg.train.epochs}] Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

    # --- モデル保存 ---
    save_path = to_absolute_path(cfg.output.model_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'epoch': cfg.train.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': OmegaConf.to_container(cfg)
    }, save_path)
    print(f"Training complete. Checkpoint saved to {save_path}")

    writer.close()

if __name__ == "__main__":
    train()