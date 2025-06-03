import torch
from torch.utils.data import Dataset
from MahjongGB import MahjongShanten

from pymj.botzone.action import *
from pymj.botzone.GameData import GameData
from pymj.game.chinese_ofiicial_mahjiong.Card import Card
from pymj.agent.chinese_official_mahjiong.DQNAgent import Network


class SLDataset(Dataset):
    # 天地人胡（对于模型训练没有意义）和包含连杠（数据处理模块尚未实现连杠处理）的比赛
    EXCLUDE_HAND_IDS = ["6162a99e5ddc087351c67f88.txt", "616264085ddc087351c5a045.txt", "616038c35ddc087351c06a04.txt",
                        "61633c0b5ddc087351c733de.txt", "61622e045ddc087351c50f86.txt", "6163f3555ddc087351c9a0ff.txt",
                        "616041ff5ddc087351c0891b.txt", "61619c3a5ddc087351c362cc.txt", "616200685ddc087351c443d7.txt",
                        "616323675ddc087351c6e47b.txt", "616406575ddc087351c9de85.txt", "616213135ddc087351c4ba1e.txt",
                        "61608ded5ddc087351c17b16.txt", "61614cba5ddc087351c266f8.txt", "616045615ddc087351c09324.txt",
                        "61616ed25ddc087351c2d1cd.txt", "6161d7d95ddc087351c3c05e.txt", "61617ed75ddc087351c30436.txt",
                        "6162021f5ddc087351c44916.txt", "61632da45ddc087351c70405.txt", "6163fe6c5ddc087351c9c64d.txt",
                        "616177bf5ddc087351c2eef5.txt", "616377505ddc087351c82da5.txt", "61616ed25ddc087351c2d1be.txt",
                        "616048bf5ddc087351c09e0a.txt", "61626fef5ddc087351c5c8e8.txt", "6163782c5ddc087351c83027.txt",
                        "6162a7e05ddc087351c67b85.txt", "6163e1125ddc087351c966b2.txt", "6163825a5ddc087351c85106.txt",
                        "6163efeb5ddc087351c9953a.txt", "616343c65ddc087351c74c7b.txt", "61604c295ddc087351c0ab27.txt",
                        "6163e80a5ddc087351c97d4f.txt", "616338af5ddc087351c72947.txt", "616191ef5ddc087351c3413f.txt",
                        "616059655ddc087351c0d400.txt", "6161d62c5ddc087351c3bb11.txt", "6162632e5ddc087351c59f3b.txt",
                        "61609a875ddc087351c1a4ab.txt", "616334685ddc087351c71aae.txt", "6163e48c5ddc087351c970c0.txt",
                        "6163ec725ddc087351c98afc.txt", "6163dbd65ddc087351c95712.txt", "61604a5e5ddc087351c0a404.txt",
                        "6161ecd95ddc087351c404e8.txt", "61634c555ddc087351c768fe.txt", "6162b5915ddc087351c6a59c.txt",
                        "6162556a5ddc087351c57314.txt", "616222b45ddc087351c4ed9d.txt", "616092f75ddc087351c18ab8.txt",
                        "6162a99e5ddc087351c67f97.txt", "6163dd9d5ddc087351c95a8c.txt", "616152cd5ddc087351c279e2.txt",
                        "616362bb5ddc087351c7ea21.txt", "6163fb025ddc087351c9b939.txt", "616367075ddc087351c7f77e.txt",
                        "6163a3795ddc087351c8a0cb.txt", "616064215ddc087351c0f4de.txt", "616358845ddc087351c78dd3.txt",
                        "6162366c5ddc087351c52be6.txt", "6163fd825ddc087351c9c35f.txt", "6161ea565ddc087351c3fb6f.txt",
                        "6162116d5ddc087351c4b553.txt", "616403a95ddc087351c9d596.txt", "6161814c5ddc087351c30bd9.txt",
                        "61619d1b5ddc087351c365d0.txt", "6161abbf5ddc087351c3950d.txt", "61605eb35ddc087351c0e433.txt",
                        "616070725ddc087351c11d63.txt", "6161acb25ddc087351c39808.txt", "6162246c5ddc087351c4f321.txt",
                        "616048bf5ddc087351c09f09.txt", "6163a1b45ddc087351c89a78.txt", "616275ed5ddc087351c5da6c.txt",
                        "6161872e5ddc087351c31f3c.txt", "6163fb025ddc087351c9babf.txt", "616030695ddc087351c0505d.txt",
                        "6161435d5ddc087351c24994.txt", "61603d005ddc087351c07875.txt", "61637fda5ddc087351c846e1.txt",
                        "616089af5ddc087351c16d60.txt", "61629aed5ddc087351c651c4.txt", "61614d955ddc087351c26a3d.txt",
                        "616085835ddc087351c15ed0.txt", "6163a29b5ddc087351c89d33.txt", "616229925ddc087351c502e7.txt",
                        "61613e3e5ddc087351c2373d.txt", "616040505ddc087351c083d0.txt", "6160321d5ddc087351c0545a.txt",
                        "61604c295ddc087351c0a9a1.txt", "6163e3a65ddc087351c96f80.txt", "6161e54e5ddc087351c3ebf4.txt",
                        "61632b145ddc087351c6fdf3.txt", "61625a995ddc087351c5840e.txt", "6161f5365ddc087351c420bb.txt",
                        "616257ef5ddc087351c57bdf.txt", "616360205ddc087351c7a677.txt", "61617fa65ddc087351c30757.txt",
                        "616387885ddc087351c86084.txt", "616406575ddc087351c9dd77.txt", "616039a25ddc087351c06f21.txt",
                        "6161f3855ddc087351c41a9b.txt", "61609a875ddc087351c1a4dd.txt", "616188de5ddc087351c3268e.txt",
                        "61620b835ddc087351c46709.txt", "616347495ddc087351c7584f.txt", "616230945ddc087351c519a1.txt",
                        "61619d1b5ddc087351c365a3.txt", "6163dd9d5ddc087351c95a1e.txt", "6161a5ad5ddc087351c38275.txt",
                        "616168ba5ddc087351c2be97.txt", "6161f96b5ddc087351c42c9e.txt", "616037e65ddc087351c068e4.txt",
                        "616176d35ddc087351c2ec6e.txt", "61618abb5ddc087351c32ad0.txt", "6161f2b65ddc087351c417ce.txt",
                        "6163e1e85ddc087351c96a54.txt", "6161db3a5ddc087351c3ca80.txt", "616035625ddc087351c05ff8.txt",
                        "616191ef5ddc087351c34261.txt", "61602df05ddc087351c047cf.txt", "616287255ddc087351c611ff.txt",
                        "616370965ddc087351c8164d.txt", "6160783c5ddc087351c13604.txt", "616040505ddc087351c0838a.txt",
                        "6163c2105ddc087351c902e0.txt", "61626d7f5ddc087351c5c0ba.txt", "61602df05ddc087351c04815.txt",
                        "616177bf5ddc087351c2ef54.txt", "6161954f5ddc087351c34d3f.txt", "6162afa05ddc087351c6943d.txt",
                        "616367f15ddc087351c7f9e3.txt", "616041265ddc087351c085c2.txt"]

    def __init__(self, data_path, device="cpu"):
        super().__init__()
        self.device = device
        data = torch.load(data_path, weights_only=True)
        self.mlp_features = data["mlp_features"].to(self.device)
        self.cnn_features = data["cnn_features"].to(self.device)
        self.rnn_features = data["rnn_features"].to(self.device)
        self.rnn_seqs_len = data["rnn_seqs_len"].to(self.device)
        self.labels = data["labels"].to(self.device)
        self.weights = data["weights"].to(self.device)
        if "mask" not in data:
            num_sample = self.labels.shape[0]
            self.mask = torch.ones((num_sample, )).long().to(self.device)
        else:
            self.mask = data["mask"]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        return (self.mlp_features[item], self.cnn_features[item], self.rnn_features[item], self.rnn_seqs_len[item],
                self.labels[item], self.weights[item], self.mask[item])

    @staticmethod
    def save_data(original_data_dir, mode, save_data_dir, num_game=10):
        assert mode in ["Chi", "Peng", "Gang", "Play"]
        data_path_names = os.listdir(original_data_dir)
        all_mlp_features = []
        all_cnn_features = []
        all_rnn_features = []
        all_labels = []
        all_weights = []
        all_gang_mask = []
        all_rnn_seqs_len = []
        n = 0
        if num_game <= 0:
            num_game = len(data_path_names)
        for data_path_name in data_path_names:
            if n >= num_game:
                break
            if ".txt" in data_path_name and data_path_name not in SLDataset.EXCLUDE_HAND_IDS:
                data_path = os.path.join(original_data_dir, data_path_name)
                hand_data = GameData.read_one_game(data_path)
                state_sequences, action_sequences = hand_data.get_state_action_sequence()
                min_seq_len = min(list(map(len, state_sequences)))
                if min_seq_len < 3:
                    # 太短的对局不要
                    continue
                player_weight = []
                for i, (state_sequence, action_sequence) in enumerate(zip(state_sequences, action_sequences)):
                    last_state = state_sequence[-1]
                    last_action = action_sequence[-1]
                    if not isinstance(last_action, (Hu, ZiMoHu)):
                        self_tiles = []
                        for card_id, card_num in last_state["self_tiles"].items():
                            for _ in range(card_num):
                                self_tiles.append(card_id)
                        open_melds = last_state["players_open_melds"][i]
                        pack = []
                        for meld_type, meld_player_id, meld_card_id in open_melds:
                            if meld_type == "Chi":
                                claiming = (meld_type.upper(), Card.decoding(meld_card_id), 1)
                            else:
                                claiming = (meld_type.replace("Bu", "").replace("An", "").upper(),
                                            Card.decoding(meld_card_id),
                                            (i - meld_player_id + 4) % 4)
                            pack.append(claiming)
                        hand = tuple(map(Card.decoding, self_tiles))
                        shanten = MahjongShanten(tuple(pack), hand)
                        player_weight.append(1 - (shanten + 1) / 10)
                        continue
                    else:
                        # 给更高的权重
                        player_weight.append(1.0)
                for player_id in range(4):
                    state_sequence = state_sequences[player_id]
                    action_sequence = action_sequences[player_id]
                    if mode in ["Chi", "Peng", "Gang"]:
                        states, actions = SLDataset.filter_open_meld_data(state_sequence, action_sequence, mode)
                    else:
                        states, actions = SLDataset.filter_play_data(state_sequence, action_sequence)
                    for state, action in zip(states, actions):
                        all_weights.append(player_weight[player_id])
                        self_player_id = player_id
                        self_wind = player_id
                        game_wind = hand_data.wind
                        if mode == "Play":
                            self_tiles = state["state4discard"]["self_tiles"]
                            players_melds = state["state4discard"]["players_open_melds"]
                        else:
                            self_tiles = state["self_tiles"]
                            players_melds = state["players_open_melds"]
                        self_hand_card_ids = []
                        for card_id, card_num in self_tiles.items():
                            for _ in range(card_num):
                                self_hand_card_ids.append(card_id)
                        players_played_card_ids = []
                        for player_played_card_ids in state["players_discarded_tiles"]:
                            players_played_card_ids.append(
                                list(map(lambda x: x[1], player_played_card_ids))
                            )
                        wind_features = Network.feature_wind(self_wind, game_wind)
                        self_hand_card_features4mlp, features4cnn = Network.feature_self_cards(
                            self_hand_card_ids)
                        self_played_card_features4mlp, self_played_card_features4rnn, self_seq_len = (
                            Network.feature_played_cards(players_played_card_ids[self_player_id], 21))
                        self_melds_features = Network.feature_melds(players_melds[self_player_id])
                        other_played_card_ids = []
                        for i in range(4):
                            if i != self_player_id:
                                other_played_card_ids.extend(players_played_card_ids[i])
                        other_played_card_features4mlp, _, _ = Network.feature_played_cards(
                            other_played_card_ids, 21
                        )
                        left_player_id = (self_player_id - 1) if (self_player_id > 0) else 3
                        right_player_id = (self_player_id + 1) if (self_player_id < 3) else 0
                        across_player_id = (self_player_id + 2) if (self_player_id < 2) else (self_player_id - 2)
                        _, left_played_card_features4rnn, left_seq_len = (
                            Network.feature_played_cards(players_played_card_ids[left_player_id], 21))
                        _, right_played_card_features4rnn, right_seq_len = (
                            Network.feature_played_cards(players_played_card_ids[right_player_id], 21))
                        _, across_played_card_features4rnn, across_seq_len = (
                            Network.feature_played_cards(players_played_card_ids[across_player_id], 21))
                        remain_card_features = Network.feature_remain_cards(
                            self_player_id, self_hand_card_ids, tuple(players_played_card_ids), players_melds)
                        left_melds_features = Network.feature_melds(players_melds[left_player_id])
                        right_melds_features = Network.feature_melds(players_melds[right_player_id])
                        across_melds_features = Network.feature_melds(players_melds[across_player_id])
                        features4mlp = torch.cat(
                            (wind_features, self_hand_card_features4mlp, self_played_card_features4mlp,
                             self_melds_features, other_played_card_features4mlp, left_melds_features,
                             right_melds_features, across_melds_features, remain_card_features), dim=0
                        ).unsqueeze(dim=0)
                        features4rnn = torch.cat((
                            self_played_card_features4rnn.unsqueeze(dim=0),
                            left_played_card_features4rnn.unsqueeze(dim=0),
                            right_played_card_features4rnn.unsqueeze(dim=0),
                            across_played_card_features4rnn.unsqueeze(dim=0)
                        ), dim=0).unsqueeze(dim=0)
                        features4cnn = features4cnn.unsqueeze(dim=0)
                        rnn_seqs_len = torch.cat([
                            self_seq_len.unsqueeze(dim=0), 
                            left_seq_len.unsqueeze(dim=0), 
                            right_seq_len.unsqueeze(dim=0), 
                            across_seq_len.unsqueeze(dim=0)], dim=0)
                        all_mlp_features.append(features4mlp)
                        all_cnn_features.append(features4cnn)
                        all_rnn_features.append(features4rnn)
                        all_rnn_seqs_len.append(rnn_seqs_len.unsqueeze(dim=0))
                        if mode == "Play":
                            all_labels.append(action.tile_out)
                        elif mode == "Gang":
                            if state["can_an_gang"]:
                                all_gang_mask.append(torch.tensor([1, 0, 0, 1]).bool().unsqueeze(dim=0))
                            elif state["can_bu_gang"]:
                                all_gang_mask.append(torch.tensor([0, 1, 0, 1]).bool().unsqueeze(dim=0))
                            else:
                                all_gang_mask.append(torch.tensor([0, 0, 1, 1]).bool().unsqueeze(dim=0))
                            if isinstance(action, AnGang):
                                all_labels.append(0)
                            elif isinstance(action, BuGang):
                                all_labels.append(1)
                            elif isinstance(action, Gang):
                                all_labels.append(2)
                            else:
                                all_labels.append(3)
                        else:
                            all_labels.append(int(action != "Pass"))
            n += 1
        all_data = {
            "mlp_features": torch.cat(all_mlp_features, dim=0),  # (n, 288, )
            "cnn_features": torch.cat(all_cnn_features, dim=0),  # (n, 4, 34)
            "rnn_features": torch.cat(all_rnn_features, dim=0),  # (n, 4, 21)
            "rnn_seqs_len": torch.cat(all_rnn_seqs_len, dim=0),  # (n, 4)
            "labels": torch.tensor(all_labels).long(),  # (n, )
            "weights": torch.tensor(all_weights).float(),  # (n, )
        }
        if mode == "Gang":
            all_data["mask"] = torch.cat(all_gang_mask, dim=0)  # (n, 4)
        torch.save(all_data, os.path.join(save_data_dir, f"{mode}.pt"))

    @staticmethod
    def filter_play_data(state_sequence, action_sequence):
        filtered_states = []
        filtered_actions = []
        for state, action in zip(state_sequence, action_sequence):
            state4discard = state["state4discard"]
            if state4discard is None:
                continue
            if isinstance(action, (Play, Chi, Peng, Gang, AnGang, BuGang)):
                filtered_states.append(state)
                filtered_actions.append(action)
        return filtered_states, filtered_actions

    @staticmethod
    def filter_open_meld_data(state_sequence, action_sequence, mode):
        filtered_states = []
        filtered_actions = []
        for state, action in zip(state_sequence, action_sequence):
            if mode == "Chi":
                can_open_meld = state["can_chi"]
            elif mode == "Peng":
                can_open_meld = state["can_peng"]
            else:
                can_open_meld = state["can_gang"] or state["can_bu_gang"] or state["can_an_gang"]
            last_action = state["last_action"]
            if can_open_meld and last_action is not None:
                filtered_states.append(state)
                filtered_actions.append(action)
        return filtered_states, filtered_actions
