# train_teacher.py
import os
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.models.teacher_model import TeacherModel  # 教師モデルの定義
from src.models.LidarVAE import LidarVAE  # VAE モデルの定義
from src.data.h5_dataset import H5SequenceDataset  # データセットの定義

@hydra.main(config_path="config", config_name="train_teacher", version_base="1.2")
def train(cfg: DictConfig):
    # 設定表示
    print("\n=== CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    # デバイス
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # データロード
    dataset = H5SequenceDataset(root_dir=cfg.data.root_dir, seq_len=cfg.data.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )

    # VAE エンコーダ読み込み・凍結
    vae = LidarVAE(latent_dim=cfg.vae.latent_dim).to(device)
    ckpt = torch.load(cfg.vae.ckpt, map_location=device)
    vae.load_state_dict(ckpt)
    encoder = vae.encoder
    for p in encoder.parameters():
        p.requires_grad = False

    # TeacherModel 初期化
    model = TeacherModel(
        vae_encoder=encoder,
        latent_dim=cfg.vae.latent_dim,
        waypoint_dim=cfg.data.waypoint_dim,  # config で 10×2=20 を指定
        action_dim=cfg.model.action_dim,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
    ).to(device)

    # Optimizer & Loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr)
    criterion = nn.SmoothL1Loss()

    # 学習ループ
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0
        for scans, waypts, prev_act, true_act in dataloader:
            scans    = scans.to(device)    # (B, T, num_beams)
            waypts   = waypts.to(device)   # (B, T, waypoint_dim)
            prev_act = prev_act.to(device) # (B, T, 2)
            true_act = true_act.to(device) # (B, T, 2)

            pred = model(scans, waypts, prev_act)  # (B, T, 2)
            loss = criterion(pred, true_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * scans.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch:02d}/{cfg.train.epochs}] Loss: {avg_loss:.4f}")

    # モデル保存
    os.makedirs(os.path.dirname(cfg.output.model_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.output.model_path)
    print(f"Training complete. Model saved to {cfg.output.model_path}")

if __name__ == "__main__":
    train()
