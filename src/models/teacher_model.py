import torch
import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(
        self,
        vae_encoder,      # 学習済み VAE の encoder
        latent_dim,       # VAE の潜在次元
        waypoint_dim=20,  # flatten 後のウエイポイント次元 (例: 10点×2次元=20)
        action_dim=2,     # アクション次元
        lstm_hidden=128,  # LSTM の隠れ状態次元
        lstm_layers=1
    ):
        super().__init__()
        # 1) VAE エンコーダを固定
        self.encoder = vae_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2) LSTM: 入力は latent_dim のみ
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # 3) FC: LSTM の出力 + waypoint + prev_action を結合してアクション出力
        self.waypoint_dim = waypoint_dim
        self.action_dim = action_dim
        fc_in_dim = lstm_hidden + waypoint_dim + action_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim)
        )

    def forward(self, scans, waypoints, prev_actions, hidden=None):
        """
        scans:        Tensor (B, T, num_beams)
        waypoints:    Tensor (B, T, N, 2)  または  (B, T, waypoint_dim)
        prev_actions: Tensor (B, T, action_dim)
        hidden:       (h0, c0) optional
        """
        B, T, _ = scans.size()

        # --- 1) 各時刻 LiDAR → 潜在 z_t ---
        scans_flat = scans.view(-1, scans.size(-1))   # (B*T, num_beams)
        with torch.no_grad():
            z_flat = self.encoder.encode(scans_flat)  # (B*T, latent_dim)
        z = z_flat.view(B, T, -1)                    # (B, T, latent_dim)

        # --- 2) LSTM に z シーケンスを入力 ---
        lstm_out, _ = self.lstm(z, hidden)           # (B, T, lstm_hidden)

        # --- 3) waypoints が4次元なら flatten ---
        if waypoints.dim() == 4:
            # waypoints: (B, T, N, 2) → (B, T, N*2)
            N = waypoints.size(2)
            waypoints = waypoints.view(B, T, N * 2)

        # --- 4) 各時刻で h_t, waypoint_t, prev_action_t を結合 ---
        #    (B, T, lstm_hidden + waypoint_dim + action_dim)
        combined = torch.cat([lstm_out, waypoints, prev_actions], dim=-1)

        # --- 5) FC でアクション回帰 ---
        actions_pred = self.fc(combined)             # (B, T, action_dim)
        return actions_pred
