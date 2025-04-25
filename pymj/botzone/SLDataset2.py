import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from pymj.botzone.action import *
from pymj.botzone.GameData import GameData


class SLDataset(IterableDataset):
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

    def __init__(self, original_data_dir, save_data_dir, mode, max_discard_length=24, device="cpu"):
        super().__init__()
        self.save_data_dir = save_data_dir
        self.mode = mode
        self.max_discard_length = max_discard_length
        self.device = device
        data_path_names = os.listdir(original_data_dir)
        self.game_data_paths = []
        for data_path_name in data_path_names:
            if ".txt" in data_path_name and data_path_name not in SLDataset.EXCLUDE_HAND_IDS:
                p = os.path.join(original_data_dir, data_path_name)
                self.game_data_paths.append(p)

    def save_data(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # 单线程：全部读取
            iter_start = 0
            iter_end = len(self.game_data_paths)
        else:
            # 多线程：分配区间
            per_worker = int(len(self.game_data_paths) / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for i in range(iter_start, iter_end):
            data_path = self.game_data_paths[i]
            hand_data = GameData.read_one_game(data_path)
            state_sequences, action_sequences = hand_data.get_state_action_sequence()
            for player_id in range(4):
                state_sequence = state_sequences[player_id]
                action_sequence = action_sequences[player_id]
                if self.mode == "Discard":
                    yield from self.parse_state_action4discard(state_sequence, action_sequence, hand_data.wind,
                                                               player_id, player_id)
                elif self.mode in ["Chi", "Peng", "Gang"]:
                    yield from self.parse_state_action4open_meld(state_sequence, action_sequence, hand_data.wind,
                                                                 player_id)
                elif self.mode in ["AnGang", "BuGang"]:
                    yield from self.parse_state_action4self_gang(state_sequence, action_sequence, hand_data.wind,
                                                                 player_id)

    @staticmethod
    def feature_self_tiles(self_tiles):
        # 手牌, 4 * 34
        self_tiles_feature = np.zeros((4, 34), dtype=np.float32)
        for tile_int, tile_num in self_tiles.items():
            for i in range(tile_num):
                self_tiles_feature[i][tile_int] = 1
        return self_tiles_feature

    @staticmethod
    def feature_discard(players_discarded_tiles, max_discard_length):
        # 四家舍牌, 4 * max_discard_length * 34
        discard_feature = []
        for played_tiles in players_discarded_tiles:
            played_tiles_feature_i = [[0] * 34 for _ in range(max_discard_length)]
            for i, (_, played_tile) in enumerate(played_tiles):
                played_tiles_feature_i[i][played_tile] = 1
            discard_feature.extend(played_tiles_feature_i)
        return np.array(discard_feature, dtype=np.float32)

    @staticmethod
    def feature_open_meld(players_open_melds):
        open_meld_feature = []
        for open_melds in players_open_melds:
            # 5种副露，吃、碰、杠、补杠、暗杠
            open_meld_feature_i = [[0] * 34 for _ in range(20)]
            for n, (open_meld_type, open_meld_tile) in enumerate(open_melds):
                if open_meld_type == "Chi":
                    k = n
                elif open_meld_type == "Peng":
                    k = 4 + n
                elif open_meld_type == "Gang":
                    k = 8 + n
                elif open_meld_type == "BuGang":
                    k = 12 + n
                else:
                    k = 16 + n
                open_meld_feature_i[k][open_meld_tile] = 1
            open_meld_feature.extend(open_meld_feature_i)
        return np.array(open_meld_feature, dtype=np.float32)

    @staticmethod
    def feature_remain_tiles(self_tiles, players_open_melds, players_discarded_tiles):
        remain_tiles_feature = np.ones((4, 34), dtype=np.float32)
        exist_tiles = {i: 0 for i in range(34)}
        for open_melds in players_open_melds:
            for _, (open_meld_type, open_meld_tile) in enumerate(open_melds):
                if open_meld_type == "Chi":
                    chi_tiles = (open_meld_tile - 1, open_meld_tile, open_meld_tile + 1)
                    for chi_tile in chi_tiles:
                        remain_tiles_feature[exist_tiles[chi_tile]][chi_tile] = 0
                        exist_tiles[chi_tile] += 1
                else:
                    for i in range(3):
                        remain_tiles_feature[i][open_meld_tile] = 0
                    if open_meld_type == "Peng":
                        exist_tiles[open_meld_tile] += 3
        for played_tiles in players_discarded_tiles:
            for _, played_tile in played_tiles:
                num_exist = exist_tiles[played_tile]
                remain_tiles_feature[num_exist - 1][played_tile] = 0
                exist_tiles[played_tile] += 1
        for tile_int, tile_num in self_tiles.items():
            num_exist = exist_tiles[tile_int] + tile_num
            if num_exist < 4:
                # 如果碰的牌是绝张，或者是杠，会重复计算手牌和碰｜杠的牌
                remain_tiles_feature[num_exist - 1][tile_int] = 0
        return np.array(remain_tiles_feature, dtype=np.float32)

    @staticmethod
    def feature_wind(QF, MF):
        wind_feature = np.zeros((8, 34), dtype=np.float32)
        wind_feature[0 + QF, :] = 1
        wind_feature[4 + MF, :] = 1
        return wind_feature

    def parse_state_action4discard(self, state_sequences, action_sequences, QF, MF, player_id):
        # 弃牌模型
        for state, action in zip(state_sequences, action_sequences):
            state4discard = state["state4discard"]
            if state4discard is None:
                continue
            if isinstance(action, (Play, Chi, Peng, Gang, AnGang, BuGang)):
                last_action = state["last_action"]
                self_tiles = state["state4discard"]["self_tiles"]
                players_open_melds = state["state4discard"]["players_open_melds"]
                players_discarded_tiles = state["players_discarded_tiles"]
                # 手牌, 4 * 34
                self_tiles_feature = self.feature_self_tiles(self_tiles)
                # 四家舍牌, 4 * max_discard_length * 34
                discard_feature = self.feature_discard(players_discarded_tiles, self.max_discard_length)
                # 四家副露, 80 * 34
                open_meld_feature = self.feature_open_meld(players_open_melds)
                # 剩余可能牌, 4 * 34
                remain_tiles_feature = self.feature_remain_tiles(
                    self_tiles, players_open_melds, players_discarded_tiles)
                # 自风、场风, 8 * 34
                wind_feature = self.feature_wind(QF, MF)
                # 检查是否有错误
                num_self_tile = self_tiles_feature.sum()
                num_open_meld = len(players_open_melds[player_id])
                assert num_self_tile == (14 - num_open_meld * 3), \
                    f"对于弃牌模型，手牌必须是14-n*3张，其中n为已有副露数量，此时手牌数为{num_self_tile}，副露数为{num_open_meld}"
                assert action.tile_out < 34, f"弃牌模型的输出必须小于34"
                # 摸牌来源：Play, Chi, Peng, Gang, AnGang, BuGang
                draw_tile_source_feature = np.zeros((6, 34), dtype=np.float32)
                if last_action is not None:
                    draw_tile_source_feature[SLDataset.DRAW_SOURCE[type(last_action)], :] = 1
                feature = np.concatenate((
                    self_tiles_feature,
                    open_meld_feature,
                    discard_feature,
                    remain_tiles_feature,
                    wind_feature,
                    draw_tile_source_feature,
                ), axis=0)
                # value in [0, 1, 2, ..., 33]
                label = action.tile_out
                yield torch.from_numpy(feature).to(self.device), torch.tensor(label).to(self.device)

    def parse_state_action4open_meld(self, state_sequences, action_sequences, QF, MF):
        # 吃碰杠模型
        for state, action in zip(state_sequences, action_sequences):
            if self.mode == "Chi":
                can_open_meld = state["can_chi"]
            elif self.mode == "Peng":
                can_open_meld = state["can_peng"]
            else:
                can_open_meld = state["can_gang"]
            last_action = state["last_action"]
            if can_open_meld and last_action is not None:
                last_tile = last_action.tile_out
                self_tiles = state["self_tiles"]
                players_open_melds = state["players_open_melds"]
                players_discarded_tiles = state["players_discarded_tiles"]
                # 手牌, 4 * 34
                self_tiles_feature = self.feature_self_tiles(self_tiles)
                # 四家舍牌, 4 * max_discard_length * 34
                discard_feature = self.feature_discard(players_discarded_tiles, self.max_discard_length)
                # 四家副露, 80 * 34
                open_meld_feature = self.feature_open_meld(players_open_melds)
                # 剩余可能牌, 4 * 34
                remain_tiles_feature = self.feature_remain_tiles(
                    self_tiles, players_open_melds, players_discarded_tiles)
                # 自风、场风, 8 * 34
                wind_feature = self.feature_wind(QF, MF)
                # 其它玩家打出的牌
                last_tile_feature = np.zeros((1, 34), dtype=np.float32)
                last_tile_feature[0, last_tile] = 1
                # 摸牌来源：Play, Chi, Peng, Gang, AnGang, BuGang
                draw_tile_source_feature = np.zeros((6, 34), dtype=np.float32)
                if last_action is not None:
                    draw_tile_source_feature[SLDataset.DRAW_SOURCE[type(last_action)], :] = 1
                feature = np.concatenate((
                    self_tiles_feature,
                    open_meld_feature,
                    discard_feature,
                    remain_tiles_feature,
                    wind_feature,
                    draw_tile_source_feature,
                    last_tile_feature
                ), axis=0)
                if action == "Pass":
                    label = 0
                else:
                    label = 1
                yield torch.from_numpy(feature).to(self.device), torch.tensor(label).to(self.device)

    def parse_state_action4self_gang(self, state_sequences, action_sequences, QF, MF):
        # 暗杠补杠模型
        for state, action in zip(state_sequences, action_sequences):
            can_open_meld = state["can_bu_gang"] if self.mode == "BuGang" else state["can_an_gang"]
            last_action = state["last_action"]
            if can_open_meld:
                tiles_can_gang = state["tiles_can_bu_gang"] if self.mode == "BuGang" else state["tiles_can_an_gang"]
                self_tiles = state["self_tiles"]
                players_open_melds = state["players_open_melds"]
                players_discarded_tiles = state["players_discarded_tiles"]
                # 手牌, 4 * 34
                self_tiles_feature = self.feature_self_tiles(self_tiles)
                # 四家舍牌, 4 * max_discard_length * 34
                discard_feature = self.feature_discard(players_discarded_tiles, self.max_discard_length)
                # 四家副露, 80 * 34
                open_meld_feature = self.feature_open_meld(players_open_melds)
                # 剩余可能牌, 4 * 34
                remain_tiles_feature = self.feature_remain_tiles(
                    self_tiles, players_open_melds, players_discarded_tiles)
                # 自风、场风, 8 * 34
                wind_feature = self.feature_wind(QF, MF)
                # 可以杠的牌
                tiles_can_gang_feature = np.zeros((1, 34), dtype=np.float32)
                for tile_can_gang in tiles_can_gang:
                    tiles_can_gang_feature[0, tile_can_gang] = 1
                feature = np.concatenate((
                    self_tiles_feature,
                    open_meld_feature,
                    discard_feature,
                    remain_tiles_feature,
                    wind_feature,
                    tiles_can_gang_feature,
                ), axis=0)
                if self.mode == "AnGang":
                    label = int(isinstance(action, AnGang))
                else:
                    label = int(isinstance(action, BuGang))
                yield torch.from_numpy(feature).to(self.device), torch.tensor(label).to(self.device)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # 单线程：全部读取
            iter_start = 0
            iter_end = len(self.game_data_paths)
        else:
            # 多线程：分配区间
            per_worker = int(len(self.game_data_paths) / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for i in range(iter_start, iter_end):
            data_path = self.game_data_paths[i]
            hand_data = GameData.read_one_game(data_path)
            state_sequences, action_sequences = hand_data.get_state_action_sequence()
            for player_id in range(4):
                state_sequence = state_sequences[player_id]
                action_sequence = action_sequences[player_id]
                if self.mode == "Discard":
                    yield from self.parse_state_action4discard(state_sequence, action_sequence, hand_data.wind, player_id, player_id)
                elif self.mode in ["Chi", "Peng", "Gang"]:
                    yield from self.parse_state_action4open_meld(state_sequence, action_sequence, hand_data.wind, player_id)
                elif self.mode in ["AnGang", "BuGang"]:
                    yield from self.parse_state_action4self_gang(state_sequence, action_sequence, hand_data.wind, player_id)
