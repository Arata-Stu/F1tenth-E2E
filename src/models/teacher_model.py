import torch
import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(
        self,
        vae_encoder,      # 学習済み LidarVAE の encoder 部分
        latent_dim,       # VAE の潜在次元
        waypoint_dim=30,  # 10点×(x,y,速度)=30
        action_dim=2,     # 出力次元
        lstm_hidden=128,
        lstm_layers=1
    ):
        super().__init__()
        # VAE エンコーダを固定
        self.encoder = vae_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # LSTM: 入力は latent_dim のみ
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # FC: LSTM出力 + waypoint_dim + action_dim を結合
        fc_in = lstm_hidden + waypoint_dim + action_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim)
        )

    def forward(self, scans, waypoints, prev_actions, hidden=None):
        """
        scans:        Tensor (B, T, num_beams)
        waypoints:    Tensor (B, T, 30)
        prev_actions: Tensor (B, T, 2)
        """
        B, T, _ = scans.size()
        # 1) flatten scans → (B*T, num_beams)
        scans_flat = scans.view(-1, scans.size(-1))

        # 2) VAE encode → mu, logvar
        with torch.no_grad():
            mu, logvar = self.encoder.encode(scans_flat)
            # reparameterize で z_flat を得る
            z_flat = self.encoder.reparameterize(mu, logvar)
        # (B*T, latent_dim) → (B, T, latent_dim)
        z = z_flat.view(B, T, -1)

        # 3) LSTM に通す
        lstm_out, _ = self.lstm(z, hidden)  # (B, T, lstm_hidden)

        # 4) 各ステップで結合
        #    lstm_out:   (B, T, lstm_hidden)
        #    waypoints:  (B, T, waypoint_dim)
        #    prev_actions:(B, T, action_dim)
        combined = torch.cat([lstm_out, waypoints, prev_actions], dim=-1)

        # 5) FC でアクション予測
        actions_pred = self.fc(combined)    # (B, T, action_dim)
        return actions_pred
