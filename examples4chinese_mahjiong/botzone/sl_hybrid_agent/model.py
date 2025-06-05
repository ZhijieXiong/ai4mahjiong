import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)


class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 若输入输出通道不同，使用1x1卷积匹配维度
        self.skip = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class DeepNetwork(nn.Module):
    def __init__(self, num_class, dim_mlp, dim_rnn, dropout=0.1):
        super().__init__()
        self.embed_played_card = nn.Embedding(35, dim_rnn)

        # MLP 分支：deep + BN + Residual
        self.mlp_branch = nn.Sequential(
            nn.Linear(dim_mlp, 256),
            ResidualMLPBlock(256),
            ResidualMLPBlock(256),
            nn.Linear(256, 128),
        )

        # CNN 分支：deep + BN + Residual
        self.cnn_branch = nn.Sequential(
            ResidualCNNBlock(1, 8),
            ResidualCNNBlock(8, 16),
            nn.Flatten(),  # (batch, 16 * 4 * 34)
            nn.Linear(16 * 4 * 34, 128),
        )

        # RNN 分支
        self.rnn = nn.GRU(dim_rnn, 64)
        self.rnn_fc = nn.Linear(64 * 4, 128)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_class)
        )

    def forward(self, mlp_features, cnn_features, rnn_features, rnn_seqs_len, label_mask=None):
        mlp_out = self.mlp_branch(mlp_features)  # (batch, 128)
        cnn_out = self.cnn_branch(cnn_features.unsqueeze(1))  # (batch, 128)
        rnn_input = self.embed_played_card(rnn_features)
        batch_size, n, seq_len, _ = rnn_input.shape
        merged_input = rnn_input.reshape(batch_size, n * seq_len, -1)
        merged_output, _ = self.rnn(merged_input)
        rnn_out = merged_output.reshape(batch_size, n, seq_len, -1)

        # gather 最后一个有效时间步的输出
        try:
            # 形状 (bs, n, 1, dim)
            expanded_indices = torch.clamp_min(
                rnn_seqs_len - 1, 0).unsqueeze(-1).expand(-1, -1, rnn_out.shape[-1]).unsqueeze(2)
        except:
            expanded_indices = torch.clamp_min(
                rnn_seqs_len - 1, 0).unsqueeze(1).unsqueeze(2).unsqueeze(0).expand(-1, -1, -1, rnn_out.shape[-1])
        rnn_out = torch.gather(rnn_out, dim=2, index=expanded_indices).squeeze(2)
        fused_features = torch.cat([mlp_out, cnn_out, self.rnn_fc(rnn_out.reshape(batch_size, -1))],
                                   dim=-1)  # (batch, 384)
        output = self.fusion(fused_features)  # (batch, num_class)

        if label_mask is not None:
            output = output.masked_fill(~label_mask, -1e6)

        return output


def load_model(model_path, model_type, device):
    assert model_type in ["Play", "Chi", "Peng", "Gang"]
    if model_type == "Play":
        model = DeepNetwork(34, 288, 64)
    elif model_type == "Gang":
        model = DeepNetwork(4, 288, 64)
    else:
        model = DeepNetwork(1, 288, 64)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")["state_dict"]
    )
    return model.to(device)
