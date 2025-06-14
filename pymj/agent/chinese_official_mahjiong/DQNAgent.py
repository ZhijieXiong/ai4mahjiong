import torch
import random
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


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
    def __init__(self, num_class, dim_mlp, dim_rnn, dropout=0.1, pretrain_path=None):
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

        if pretrain_path is not None:
            pretrained = torch.load(pretrain_path, map_location="cpu", weights_only=True)["state_dict"]
            filtered_state_dict = {k: v for k, v in pretrained.items() if not k.startswith("fusion.")}
            self.load_state_dict(filtered_state_dict, strict=False)
            freeze_module(self.mlp_branch)
            freeze_module(self.cnn_branch)
            freeze_module(self.rnn)

    def forward(self, mlp_features, cnn_features, rnn_features, rnn_seqs_len, label_mask=None, noise=False):
        if self.training and noise:
            mlp_features = mlp_features * (1 + torch.randn_like(mlp_features) * 0.1)
            cnn_features = cnn_features * (1 + torch.randn_like(cnn_features) * 0.05)

        mlp_out = self.mlp_branch(mlp_features)  # (batch, 128)

        cnn_out = self.cnn_branch(cnn_features.unsqueeze(1))  # (batch, 128)

        rnn_input = self.embed_played_card(rnn_features)
        batch_size, n, seq_len, _ = rnn_input.shape
        merged_input = rnn_input.reshape(batch_size, n * seq_len, -1)
        merged_output, _ = self.rnn(merged_input)
        rnn_out = merged_output.reshape(batch_size, n, seq_len, -1)

        # gather 最后一个有效时间步的输出
        # expanded_indices = torch.clamp_min(rnn_seqs_len - 1, 0).unsqueeze(1).unsqueeze(2).expand(-1, n, rnn_out.shape[-1])
        try:
            # 形状 (bs, n, 1, dim)
            expanded_indices = torch.clamp_min(rnn_seqs_len - 1, 0).unsqueeze(-1).expand(-1, -1,
                                                                                         rnn_out.shape[-1]).unsqueeze(2)
        except:
            expanded_indices = torch.clamp_min(rnn_seqs_len - 1, 0).unsqueeze(1).unsqueeze(2).unsqueeze(0).expand(-1,
                                                                                                                  -1,
                                                                                                                  -1,
                                                                                                                  rnn_out.shape[
                                                                                                                      -1])
        rnn_out = torch.gather(rnn_out, dim=2, index=expanded_indices).squeeze(2)
        fused_features = torch.cat([mlp_out, cnn_out, self.rnn_fc(rnn_out.reshape(batch_size, -1))],
                                   dim=-1)  # (batch, 384)
        output = self.fusion(fused_features)  # (batch, num_class)

        if label_mask is not None:
            output = output.masked_fill(~label_mask, -1e6)

        return output


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


class Network(nn.Module):
    def __init__(self, num_class, dim_mlp, dim_rnn, dropout=0.1, pretrain_path=None):
        super().__init__()
        self.embed_played_card = nn.Embedding(35, dim_rnn)

        # MLP 分支
        self.mlp_branch = nn.Sequential(
            nn.Linear(dim_mlp, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )

        # CNN 分支
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(8, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Flatten(),  # 输出为 batch × (16 * 4 * 34)
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

        if pretrain_path is not None:
            pretrained = torch.load(pretrain_path, map_location="cpu", weights_only=True)["state_dict"]
            filtered_state_dict = {k: v for k, v in pretrained.items() if not k.startswith("fusion.")}
            self.load_state_dict(filtered_state_dict, strict=False)
            freeze_module(self.mlp_branch)
            freeze_module(self.cnn_branch)
            freeze_module(self.rnn)

    def forward(self, mlp_features, cnn_features, rnn_features, rnn_seqs_len, label_mask=None, noise=False):
        if self.training and noise:
            mlp_features = mlp_features * (1 + torch.randn_like(mlp_features) * 0.1)
            cnn_features = cnn_features * (1 + torch.randn_like(cnn_features) * 0.05)

        mlp_out = self.mlp_branch(mlp_features)  # (batch, 128)
        cnn_out = self.cnn_branch(cnn_features.unsqueeze(1))  # (batch, 128)
        rnn_input = self.embed_played_card(rnn_features)
        batch_size, n, seq_len, _ = rnn_input.shape
        merged_input = rnn_input.reshape(batch_size, n * seq_len, -1)
        merged_output, _ = self.rnn(merged_input)
        rnn_out = merged_output.reshape(batch_size, n, seq_len, -1)  # (batch, 4, 64)
        try:
            # 形状 (bs, n, 1, dim)
            expanded_indices = torch.clamp_min(rnn_seqs_len - 1, 0).unsqueeze(-1).expand(-1, -1,
                                                                                         rnn_out.shape[-1]).unsqueeze(2)
        except:
            expanded_indices = torch.clamp_min(rnn_seqs_len - 1, 0).unsqueeze(1).unsqueeze(2).unsqueeze(0).expand(-1,
                                                                                                                  -1,
                                                                                                                  -1,
                                                                                                                  rnn_out.shape[
                                                                                                                      -1])
        rnn_out = torch.gather(rnn_out, dim=2, index=expanded_indices).squeeze(2)
        fused_features = torch.cat([mlp_out, cnn_out, self.rnn_fc(rnn_out.reshape(batch_size, -1))],
                                   dim=-1)  # (batch, 384)
        output = self.fusion(fused_features)  # (batch, 34)

        if label_mask is not None:
            output = output.masked_fill(~label_mask, -1e6)

        return output

    @staticmethod
    def feature_self_cards(self_hand_card_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        exist_card_num: dict = defaultdict(int)
        features4mlp: list[int] = [0] * 34
        features4cnn: list[list[int]] = [[0] * 34 for _ in range(4)]
        for card_id in self_hand_card_ids:
            features4mlp[card_id] += 1
            features4cnn[exist_card_num[card_id]][card_id] = 1
            exist_card_num[card_id] += 1
        return torch.tensor(features4mlp).float(), torch.tensor(features4cnn).float()

    @staticmethod
    def feature_melds(melds: list[tuple[str, int, int]]) -> torch.Tensor:
        melds_features = []
        for meld_type, _, meld_card_id in melds:
            meld_features = [0] * 5 + [-1] * 4
            if meld_type == "Chi":
                meld_features[0] = 1
                meld_features[5] = meld_card_id - 1
                meld_features[6] = meld_card_id
                meld_features[7] = meld_card_id + 1
            elif meld_type == "Peng":
                meld_features[1] = 1
                meld_features[5:8] = [meld_card_id] * 3
            else:
                if meld_type == "Gang":
                    meld_features[2] = 1
                elif meld_type == "AnGang":
                    meld_features[3] = 1
                else:
                    meld_features[4] = 1
                meld_features[5:9] = [meld_card_id] * 4
            melds_features.append(meld_features)
        while len(melds_features) < 4:
            melds_features.append([0] * 5 + [-1] * 4)
        random.shuffle(melds_features)
        return torch.flatten(torch.tensor(melds_features)).float()

    @staticmethod
    def feature_played_cards(
            played_cards: list[int], num_history: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padding_len = max(0, num_history - len(played_cards))
        features4rnn = [card_id + 1 for card_id in played_cards[-num_history:]] + [0] * padding_len
        features4mlp: list[int] = [0] * 34
        for card_id in played_cards:
            features4mlp[card_id] += 1
        return (torch.tensor(features4mlp).float(),
                torch.tensor(features4rnn).long(),
                torch.tensor(min(num_history, len(played_cards))).long())

    @staticmethod
    def feature_wind(self_wind: int, game_wind: int) -> torch.Tensor:
        features = [0] * 8
        features[self_wind] = 1
        features[4 + game_wind] = 1
        return torch.tensor(features).long()

    @staticmethod
    def feature_remain_cards(
            self_player_id: int,
            self_hand_card_ids: list[int],
            players_played_card_ids: tuple[list[int], ...],
            players_melds: tuple[list[tuple[str, int, int]], ...]
    ) -> torch.Tensor:
        features: list[int] = [0] * 34
        exist_tiles = {i: 4 for i in range(34)}
        for player_id, player_melds in enumerate(players_melds):
            for meld_type, _, meld_card_id in player_melds:
                if meld_type == "Chi":
                    exist_tiles[meld_card_id - 1] -= 1
                    exist_tiles[meld_card_id] -= 1
                    exist_tiles[meld_card_id + 1] -= 1
                elif meld_type == "Peng":
                    exist_tiles[meld_card_id] -= 3
                elif meld_type in ["Gang", "BuGang"]:
                    exist_tiles[meld_card_id] -= 4
                else:
                    if player_id == self_player_id:
                        exist_tiles[meld_card_id] -= 4
        for player_played_card_ids in players_played_card_ids:
            for player_played_card_id in player_played_card_ids:
                exist_tiles[player_played_card_id] -= 1
        for card_id in self_hand_card_ids:
            exist_tiles[card_id] -= 1
        for card_id, num_card in exist_tiles.items():
            for _ in range(num_card):
                features[card_id] += 1
        return torch.tensor(features).float()

    @staticmethod
    def get_features(state: dict, num_history) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_player_id: int = state["self_player_id"]
        self_wind: int = state["self_player_id"]
        game_wind: int = state["game_wind"]
        self_hand_card_ids: list[int] = state["self_hand_card_ids"]
        players_played_card_ids: tuple[list[int], ...] = state["players_played_card_ids"]
        players_melds: tuple[list[tuple[str, int, int]], ...] = state["players_melds"]

        wind_features = Network.feature_wind(self_wind, game_wind)
        self_hand_card_features4mlp, features4cnn = Network.feature_self_cards(self_hand_card_ids)
        self_played_card_features4mlp, self_played_card_features4rnn, self_seq_len = Network.feature_played_cards(
            players_played_card_ids[self_player_id], num_history
        )
        self_melds_features = Network.feature_melds(players_melds[self_player_id])
        other_played_card_ids = []
        for i in range(4):
            if i != self_player_id:
                other_played_card_ids.extend(players_played_card_ids[i])
        other_played_card_features4mlp, _, _ = Network.feature_played_cards(other_played_card_ids, num_history)
        left_player_id = (self_player_id - 1) if (self_player_id > 0) else 3
        right_player_id = (self_player_id + 1) if (self_player_id < 3) else 0
        across_player_id = (self_player_id + 2) if (self_player_id < 2) else (self_player_id - 2)
        _, left_played_card_features4rnn, left_seq_len = (
            Network.feature_played_cards(players_played_card_ids[left_player_id], num_history))
        _, right_played_card_features4rnn, right_seq_len = (
            Network.feature_played_cards(players_played_card_ids[right_player_id], num_history))
        _, across_played_card_features4rnn, across_seq_len = (
            Network.feature_played_cards(players_played_card_ids[across_player_id], num_history))
        remain_card_features = Network.feature_remain_cards(
            self_player_id, self_hand_card_ids, players_played_card_ids, players_melds)
        left_melds_features = Network.feature_melds(players_melds[left_player_id])
        right_melds_features = Network.feature_melds(players_melds[right_player_id])
        across_melds_features = Network.feature_melds(players_melds[across_player_id])
        features4mlp = torch.cat(
            (wind_features, self_hand_card_features4mlp, self_played_card_features4mlp,
             self_melds_features, other_played_card_features4mlp, left_melds_features,
             right_melds_features, across_melds_features, remain_card_features), dim=0
        )
        features4rnn = torch.cat((
            self_played_card_features4rnn.unsqueeze(dim=0), left_played_card_features4rnn.unsqueeze(dim=0),
            right_played_card_features4rnn.unsqueeze(dim=0), across_played_card_features4rnn.unsqueeze(dim=0)
        ), dim=0)

        return features4mlp, features4cnn, features4rnn
