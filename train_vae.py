import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.LidarVAE import LidarVAE
from src.data.h5_dataset import H5ScansDataset

@hydra.main(config_path="config", config_name="train_vae", version_base="1.2")
def main(cfg: DictConfig):
    # 設定表示
    print(OmegaConf.to_yaml(cfg))

    # データセット & DataLoader
    dataset = H5ScansDataset(cfg.data.root_dir)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers
    )

    # モデル初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = LidarVAE(
        num_beams=cfg.model.num_beams,
        latent_dim=cfg.model.latent_dim
    ).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.train.lr)

    # TensorBoard Writer
    log_dir = cfg.train.get('log_dir', 'runs')
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    for epoch in range(cfg.train.epochs):
        vae.train()
        total_loss = 0.0
        # tqdm でプログレスバー
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}", unit="batch")
        for scans in pbar:
            scans = scans.to(device)
            recon, mu, logvar = vae(scans)
            # 再構成損失
            loss_recon = F.mse_loss(recon, scans)
            # KL損失
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_recon + cfg.train.beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * scans.size(0)

            # バッチごとにプログレスバーに表示
            pbar.set_postfix(loss=loss.item())

        # エポックごとの平均損失
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{cfg.train.epochs}, Loss: {avg_loss:.6f}")
        # TensorBoard にログ
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/recon', loss_recon.item(), epoch)
        writer.add_scalar('Loss/kl', kl.item(), epoch)

    # 学習済みモデル保存
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    save_path = os.path.join(cfg.train.ckpt_dir, 'lidar_vae.pth')
    vae.save(save_path)
    print(f"Saved VAE checkpoint to {save_path}")

    writer.close()

if __name__ == '__main__':
    main()
