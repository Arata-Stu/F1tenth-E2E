import torch
import torch.nn as nn
import torch.nn.functional as F

class LidarVAE(nn.Module):
    """
    Variational Autoencoder for 2D LiDAR scans using 1D Conv layers.
    """
    def __init__(self, lidar_dim: int = 1080, latent_dim: int = 32):
        super().__init__()
        # Encoder: convolutional feature extractor
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.pool  = nn.AdaptiveAvgPool1d(output_size=16)
        conv_feat_size = 64 * 16
        # Latent projections
        self.fc_mu     = nn.Linear(conv_feat_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_feat_size, latent_dim)
        # Decoder: project latent back to conv space
        self.fc_dec = nn.Linear(latent_dim, conv_feat_size)
        self.deconv1 = nn.ConvTranspose1d(64, 48, kernel_size=3, stride=1)
        self.deconv2 = nn.ConvTranspose1d(48, 36, kernel_size=3, stride=1)
        self.deconv3 = nn.ConvTranspose1d(36, 24, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose1d(24, 16, kernel_size=8, stride=4, padding=2)
        self.deconv5 = nn.ConvTranspose1d(16, 1,  kernel_size=10, stride=4, padding=3)

    def encode(self, x: torch.Tensor):
        # x: (batch, lidar_dim)
        x = x.unsqueeze(1)               # (batch,1,lidar_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)                 # (batch,64,16)
        x = x.view(x.size(0), -1)        # (batch, conv_feat_size)
        mu     = self.fc_mu(x)           # (batch, latent_dim)
        logvar = self.fc_logvar(x)       # (batch, latent_dim)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        x = F.relu(self.fc_dec(z))
        x = x.view(x.size(0), 64, 16)    # (batch,64,16)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))  # (batch,1,lidar_dim)
        return x.squeeze(1)              # (batch, lidar_dim)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def save(self, path: str):
        """Save model state_dict to path."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        """Load model state_dict from path."""
        state = torch.load(path, map_location=map_location) if map_location else torch.load(path)
        self.load_state_dict(state)
        return self
