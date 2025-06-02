import torch
from torch.utils.data import Dataset

from pymj.botzone.action import *
from pymj.botzone.GameData import GameData


class SLDataset(Dataset):
    ALL_TILES = {i: 4 for i in range(34)}
    DRAW_SOURCE = {
        Play: 0,
        Chi: 1,
        Peng: 2,
        Gang: 3,
        BuGang: 4,
        AnGang: 5
    }
    MELD_TYPE = {
        "Chi": 0,
        "Peng": 1,
        "Gang": 2,
        "BuGang": 3,
        "AnGang": 4
    }
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
        self.features = data["features"].to(self.device)
        self.labels = data["labels"].to(self.device)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, item):
        return self.features[item].to_dense().float(), self.labels[item]

    @staticmethod
    def save_data(original_data_dir, mode, save_data_dir):
        assert mode in ["Chi", "Peng", "Gang", "AnGang", "BuGang", "Play"]
        data_path_names = os.listdir(original_data_dir)
        all_features = []
        all_labels = []
        for data_path_name in data_path_names:
            if ".txt" in data_path_name and data_path_name not in SLDataset.EXCLUDE_HAND_IDS:
                data_path = os.path.join(original_data_dir, data_path_name)
                hand_data = GameData.read_one_game(data_path)
                state_sequences, action_sequences = hand_data.get_state_action_sequence()
                for player_id in range(4):
                    state_sequence = state_sequences[player_id]
                    action_sequence = action_sequences[player_id]
                    if mode in ["Chi", "Peng", "Gang"]:
                        features, labels = SLDataset.get_open_meld_data(state_sequence, action_sequence, hand_data.wind, player_id, player_id, mode)
                    elif mode in ["AnGang", "BuGang"]:
                        features, labels = SLDataset.get_self_gang_data(state_sequence, action_sequence, hand_data.wind, player_id, player_id, mode)
                    else:
                        features, labels = SLDataset.get_play_data(state_sequence, action_sequence, hand_data.wind, player_id, player_id)
                    if (features.shape[0] > 0) and (labels.shape[0] > 0):
                        all_features.append(features)
                        all_labels.append(labels)
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        torch.save({
            "features": features,
            "labels": labels
        }, os.path.join(save_data_dir, f"{mode}.pt"))

    @staticmethod
    def feature_self_tiles(self_tiles):
        i_self_tiles = [[0] * 35 for _ in range(4)]
        for card_id, num_card in self_tiles.items():
            for i in range(num_card):
                i_self_tiles[i][card_id] += 1
        return i_self_tiles

    @staticmethod
    def feature_played_tiles(players_discarded_tiles):
        i_players_discarded_tiles = []
        for player_discarded_tiles in players_discarded_tiles:
            i_player_discarded_tiles = [0] * 35
            for _, played_tile in player_discarded_tiles:
                i_player_discarded_tiles[played_tile] += 1
            i_players_discarded_tiles.append(i_player_discarded_tiles)
        return i_players_discarded_tiles

    @staticmethod
    def feature_open_meld(players_open_melds):
        i_players_open_melds = []
        for player_open_melds in players_open_melds:
            i_player_open_melds = []
            for meld_type, _, meld_card_id in player_open_melds:
                i_player_open_meld = [0] * 35
                i_player_open_meld[34] = SLDataset.MELD_TYPE[meld_type]
                if meld_type != "AnGang":
                    i_player_open_meld[meld_card_id] = 1
                else:
                    i_player_open_meld[34] = 1
                i_player_open_melds.append(i_player_open_meld)
            for _ in range(4 - len(player_open_melds)):
                i_player_open_melds.append([0] * 35)
            i_players_open_melds.extend(i_player_open_melds)
        return i_players_open_melds

    @staticmethod
    def feature_remain_tiles(self_tiles, players_open_melds, players_discarded_tiles):
        i_remain_tiles = [0] * 35
        exist_tiles = {i: 4 for i in range(34)}
        for open_melds in players_open_melds:
            for _, (open_meld_type, open_meld_tile) in enumerate(open_melds):
                if open_meld_type == "Chi":
                    exist_tiles[open_meld_tile - 1] -= 1
                    exist_tiles[open_meld_tile] -= 1
                    exist_tiles[open_meld_tile + 1] -= 1
                elif open_meld_type == "Peng":
                    exist_tiles[open_meld_tile] -= 3
                elif open_meld_type in ["Gang", "BuGang"]:
                    exist_tiles[open_meld_tile] -= 4
        for played_tiles in players_discarded_tiles:
            for _, played_tile in played_tiles:
                exist_tiles[played_tile] -= 1
        for tile_int, tile_num in self_tiles.items():
            exist_tiles[tile_int] -= tile_num
        for card_id, num_card in exist_tiles.items():
            for _ in range(num_card):
                i_remain_tiles[card_id] += 1
        return [i_remain_tiles]

    @staticmethod
    def feature_wind(QF, MF):
        i_wind = [0] * 35
        i_wind[QF] = 1
        i_wind[MF + 4] = 1
        return [i_wind]

    @staticmethod
    def feature_player_id(player_id):
        i_player_id = [0] * 35
        i_player_id[player_id] = 1
        return [i_player_id]

    @staticmethod
    def get_play_data(state_sequences, action_sequences, QF, MF, player_id):
        # 收集弃牌模型的数据
        inputs = []
        outputs = []
        for state, action in zip(state_sequences, action_sequences):
            state4discard = state["state4discard"]
            if state4discard is None:
                continue
            if isinstance(action, (Play, Chi, Peng, Gang, AnGang, BuGang)):
                # last_action: str
                last_action = state["last_action"]
                # self_tiles: dict[int: int]
                self_tiles = state["state4discard"]["self_tiles"]
                # players_open_melds: list[list[tuple[str, int]]]
                players_open_melds = state["state4discard"]["players_open_melds"]
                # players_discarded_tiles: list[list[tuple[_, int]]]
                players_discarded_tiles = state["players_discarded_tiles"]

                input_data = []

                # 位置：1 * 35
                input_data.extend(SLDataset.feature_player_id(player_id))

                # 手牌：4 * 35
                input_data.extend(SLDataset.feature_self_tiles(self_tiles))

                # 副露：16 * 35
                input_data.extend(SLDataset.feature_open_meld(players_open_melds))

                # 弃牌：4 * 35
                input_data.extend(SLDataset.feature_played_tiles(players_discarded_tiles))

                # 剩余牌：1 * 35
                input_data.extend(SLDataset.feature_remain_tiles(self_tiles, players_open_melds, players_discarded_tiles))

                # 风：1 *35
                input_data.extend(SLDataset.feature_wind(QF, MF))

                # 弃牌前一个动作：1 * 35
                i_draw_source = [0] * 35
                if last_action is not None:
                    i_draw_source[SLDataset.DRAW_SOURCE[type(last_action)]] = 1
                input_data.append(i_draw_source)

                inputs.append(input_data)
                outputs.append(action.tile_out)
        return torch.tensor(inputs).to_sparse(), torch.tensor(outputs)

    @staticmethod
    def get_open_meld_data(state_sequences, action_sequences, QF, MF, player_id, mode):
        # 收集副露模型的数据
        inputs = []
        outputs = []
        for state, action in zip(state_sequences, action_sequences):
            if mode == "Chi":
                can_open_meld = state["can_chi"]
            elif mode == "Peng":
                can_open_meld = state["can_peng"]
            else:
                can_open_meld = state["can_gang"]
            last_action = state["last_action"]
            if can_open_meld and last_action is not None:
                self_tiles = state["self_tiles"]
                players_open_melds = state["players_open_melds"]
                players_discarded_tiles = state["players_discarded_tiles"]

                input_data = []

                # 位置：1 * 35
                input_data.extend(SLDataset.feature_player_id(player_id))

                # 手牌：4 * 35
                input_data.extend(SLDataset.feature_self_tiles(self_tiles))

                # 副露：16 * 35
                input_data.extend(SLDataset.feature_open_meld(players_open_melds))

                # 弃牌：4 * 35
                input_data.extend(SLDataset.feature_played_tiles(players_discarded_tiles))

                # 剩余牌：1 * 35
                input_data.extend(SLDataset.feature_remain_tiles(self_tiles, players_open_melds, players_discarded_tiles))

                # 风：1 *35
                input_data.extend(SLDataset.feature_wind(QF, MF))

                # 副露的前一个动作：1 * 35
                i_draw_source = [0] * 35
                if last_action is not None:
                    if type(last_action) in SLDataset.DRAW_SOURCE:
                        i_draw_source[SLDataset.DRAW_SOURCE[type(last_action)]] = 1
                    else:
                        i_draw_source[len(SLDataset.DRAW_SOURCE)] = 1
                input_data.append(i_draw_source)

                inputs.append(input_data)
                outputs.append(int(action != "Pass"))
        return torch.tensor(inputs).to_sparse(), torch.tensor(outputs)

    @staticmethod
    def get_self_gang_data(state_sequences, action_sequences, QF, MF, player_id, mode):
        # 收集暗杠补杠模型的数据
        inputs = []
        outputs = []
        for state, action in zip(state_sequences, action_sequences):
            can_open_meld = state["can_bu_gang"] if mode == "BuGang" else state["can_an_gang"]
            if can_open_meld:
                tiles_can_gang = state["tiles_can_bu_gang"] if mode == "BuGang" else state["tiles_can_an_gang"]
                self_tiles = state["self_tiles"]
                players_open_melds = state["players_open_melds"]
                players_discarded_tiles = state["players_discarded_tiles"]

                input_data = []

                # 位置：1 * 35
                input_data.extend(SLDataset.feature_player_id(player_id))

                # 手牌：4 * 35
                input_data.extend(SLDataset.feature_self_tiles(self_tiles))

                # 副露：16 * 35
                input_data.extend(SLDataset.feature_open_meld(players_open_melds))

                # 弃牌：4 * 35
                input_data.extend(SLDataset.feature_played_tiles(players_discarded_tiles))

                # 剩余牌：1 * 35
                input_data.extend(SLDataset.feature_remain_tiles(self_tiles, players_open_melds, players_discarded_tiles))

                # 风：1 *35
                input_data.extend(SLDataset.feature_wind(QF, MF))

                # 可以杠的牌
                i_can_gang_tile = [0] * 35
                for tile_can_gang in tiles_can_gang:
                    i_can_gang_tile[tile_can_gang] = 1
                input_data.append(i_can_gang_tile)

                inputs.append(input_data)
                if mode == "AnGang":
                    outputs.append(int(isinstance(action, AnGang)))
                else:
                    outputs.append(int(isinstance(action, BuGang)))
        return torch.tensor(inputs).to_sparse(), torch.tensor(outputs)
