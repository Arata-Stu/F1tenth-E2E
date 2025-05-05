# train_teacher.py
import os
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

    # ログディレクトリに TensorBoardWriter を作成
    # Hydra による run ディレクトリに自動的に出力されます
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
    vae = LidarVAE(lidar_dim=1080, latent_dim=cfg.vae.latent_dim).to(device)
    ckpt = torch.load(cfg.vae.ckpt, map_location=device)
    vae.load_state_dict(ckpt)
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

    # --- 学習ループ ---
    global_step = 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0

        # tqdm でプログレスバー表示
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.train.epochs}", leave=False)
        for scans, waypts, prev_act, true_act in loop:
            scans    = scans.to(device)
            waypts   = waypts.to(device)
            prev_act = prev_act.to(device)
            true_act = true_act.to(device)

            # 順伝播
            pred = model(scans, waypts, prev_act)
            loss = criterion(pred, true_act)

            # 逆伝播＆更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * scans.size(0)
            global_step += 1

            # バッチ単位で TensorBoard にも記録
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            # tqdm の説明文にも現在のバッチ損失を表示
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch:02d}/{cfg.train.epochs}] Loss: {avg_loss:.4f}")

        # エポックごとに TensorBoard に記録
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

    # --- モデル保存 ---
    os.makedirs(os.path.dirname(cfg.output.model_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.output.model_path)
    print(f"Training complete. Model saved to {cfg.output.model_path}")

    writer.close()

if __name__ == "__main__":
    train()
