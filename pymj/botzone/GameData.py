from pymj.botzone.action import *


class GameData:
    # todo: 把连杠给跳过了，还没修复bug
    def __init__(self, data_lines):
        # 圈风
        self.wind = int(data_lines[0].split(" ")[1])
        # 手牌
        self.players_tiles = []
        for i in range(4):
            deal = data_lines[i+1].split(" ")[3:]
            deal_int = list(map(tile_str2int, deal))
            tiles = {i: 0 for i in range(34)}
            for tile in deal_int:
                tiles[tile] += 1
            self.players_tiles.append(tiles)
        # 副露
        self.players_open_melds = [[] for _ in range(4)]
        # 弃牌（不包括被吃、碰、杠掉的牌）
        self.players_discarded_tiles = [[] for _ in range(4)]
        # 所有的动作
        self.huang = True
        self.action_sequence = []
        i = 6
        while i < len(data_lines):
            line_last = data_lines[i - 1]
            line_this = data_lines[i]
            if "Ignore" in line_this:
                line_this = line_this.split("Ignore")[0].strip()
            if "Ignore" in line_last:
                line_last = line_last.split("Ignore")[0].strip()

            if ("Draw " in line_last) and ("Play " in line_this):
                action_player_id = int(line_last.split(" ")[1])
                tile_draw = line_last.split(" ")[3]
                tile_play = line_this.split(" ")[3]
                self.action_sequence.append((action_player_id, Play(tile_draw, tile_play)))
                i += 1
            elif ("Draw " in line_last) and ("AnGang " in line_this):
                action_player_id = int(line_last.split(" ")[1])
                line_next = data_lines[i + 1]
                line_next_ = data_lines[i + 2]
                if "Hu " in line_next_:
                    # 杠上开花
                    line_next__ = data_lines[i + 3]
                    tile_hu = line_next_.split(" ")[3]
                    fans_str = line_next__.split(" ")[2]
                    self.action_sequence.append((action_player_id, ZiMoHu(tile_hu, fans_str)))
                    self.huang = False
                    break
                tile_draw1 = line_last.split(" ")[3]
                tile_gang = line_this.split(" ")[3]
                tile_draw2 = line_next.split(" ")[3]
                tile_play = line_next_.split(" ")[3]
                self.action_sequence.append((action_player_id, AnGang(tile_draw1, tile_gang, tile_draw2, tile_play)))
                i += 3
            elif ("Draw " in line_last) and ("BuGang " in line_this):
                action_player_id = int(line_last.split(" ")[1])
                line_next = data_lines[i + 1]
                if "Hu " in line_next:
                    # 抢杠胡，国标只有抢补杠算番，抢明杠是被忽略操作，不加番，同理杠上炮也不加番，十三幺不能抢暗杠
                    hu_player_id = int(line_next.split(" ")[1])
                    line_next_ = data_lines[i + 2]
                    tile_hu = line_this.split(" ")[3]
                    fans_str = line_next_.split(" ")[2]
                    self.action_sequence.append((hu_player_id, Hu(action_player_id, tile_hu, fans_str)))
                    self.huang = False
                    break
                line_next_ = data_lines[i + 2]
                if "Hu " in line_next_:
                    # 杠上开花
                    line_next__ = data_lines[i + 3]
                    tile_hu = line_next_.split(" ")[3]
                    fans_str = line_next__.split(" ")[2]
                    self.action_sequence.append((action_player_id, ZiMoHu(tile_hu, fans_str)))
                    self.huang = False
                    break
                tile_draw1 = line_last.split(" ")[3]
                tile_gang = line_this.split(" ")[3]
                tile_draw2 = line_next.split(" ")[3]
                tile_play = line_next_.split(" ")[3]
                self.action_sequence.append((action_player_id, BuGang(tile_draw1, tile_gang, tile_draw2, tile_play)))
                i += 3
            elif ("Play " in line_last) and ("Gang " in line_this):
                last_player_id = int(line_last.split(" ")[1])
                action_player_id = int(line_this.split(" ")[1])
                line_next = data_lines[i + 1]
                line_next_ = data_lines[i + 2]
                if "Hu " in line_next_:
                    # 杠上开花
                    line_next__ = data_lines[i + 3]
                    tile_hu = line_next_.split(" ")[3]
                    fans_str = line_next__.split(" ")[2]
                    self.action_sequence.append((action_player_id, ZiMoHu(tile_hu, fans_str)))
                    self.huang = False
                    break
                tile_gang = line_this.split(" ")[3]
                tile_draw = line_next.split(" ")[3]
                tile_play = line_next_.split(" ")[3]
                self.action_sequence.append((action_player_id, Gang(last_player_id, tile_gang, tile_draw, tile_play)))
                i += 3
            elif ("Play " in line_last) and ("Peng " in line_this):
                last_player_id = int(line_last.split(" ")[1])
                action_player_id = int(line_this.split(" ")[1])
                line_next = data_lines[i + 1]
                tile_peng = line_last.split(" ")[3]
                tile_play = line_next.split(" ")[3]
                self.action_sequence.append((action_player_id, Peng(last_player_id, tile_peng, tile_play)))
                i += 1
            elif ("Play " in line_last) and ("Chi " in line_this):
                last_player_id = int(line_last.split(" ")[1])
                action_player_id = int(line_this.split(" ")[1])
                line_next = data_lines[i + 1]
                tile_chi = line_last.split(" ")[-1]
                tile_chi_middle = line_this.split(" ")[3]
                tile_play = line_next.split(" ")[3]
                self.action_sequence.append((action_player_id, Chi(last_player_id, tile_chi, tile_chi_middle, tile_play)))
                i += 1
            elif ("Draw " in line_last) and ("Hu " in line_this):
                action_player_id = int(line_this.split(" ")[1])
                line_next = data_lines[i + 1]
                tile_hu = line_last.split(" ")[3]
                fans_str = line_next.split(" ")[2]
                self.action_sequence.append((action_player_id, ZiMoHu(tile_hu, fans_str)))
                self.huang = False
                break
            elif ("Play " in line_last) and ("Hu " in line_this):
                last_player_id = int(line_last.split(" ")[1])
                action_player_id = int(line_this.split(" ")[1])
                line_next = data_lines[i + 1]
                tile_hu = line_last.split(" ")[3]
                fans_str = line_next.split(" ")[2]
                self.action_sequence.append((action_player_id, Hu(last_player_id, tile_hu, fans_str)))
                self.huang = False
                break
            else:
                i += 1

    def update_state(self, action_player_id, action):
        if isinstance(action, (Peng, Chi, Gang)):
            action.next_tiles_(self.players_tiles[action_player_id],
                               self.players_open_melds[action_player_id],
                               self.players_discarded_tiles[action.played_player_id],
                               self.players_discarded_tiles[action_player_id])
        elif isinstance(action, (BuGang, AnGang)):
            action.next_tiles_(self.players_tiles[action_player_id],
                               self.players_open_melds[action_player_id],
                               self.players_discarded_tiles[action_player_id])
        elif isinstance(action, Play):
            action.next_tiles_(self.players_tiles[action_player_id],
                               self.players_discarded_tiles[action_player_id])
        elif isinstance(action, (Hu, ZiMoHu)):
            action.next_tiles_(self.players_tiles[action_player_id])
        else:
            raise NotImplementedError()

    def get_state_action_sequence(self):
        player_ids = list(range(4))
        state_sequences = [[] for _ in player_ids]
        action_sequences = [[] for _ in player_ids]
        for action_i, (action_player_id, action) in enumerate(self.action_sequence):
            if action_i > 0:
                last_action_player_id, last_action = self.action_sequence[action_i - 1]
                self.update_state(last_action_player_id, last_action)
                for player_id in player_ids:
                    num_tile = sum(self.players_tiles[player_id].values())
                    num_open_meld = len(self.players_open_melds[player_id])
                    assert num_tile == (13 - num_open_meld * 3), \
                        f"手牌必须是13-n*3张，其中n为已有副露数量，此处手牌数为{num_tile}，n={num_open_meld}"
            else:
                last_action_player_id, last_action = None, None
            for player_id in player_ids:
                players_open_melds = deepcopy(self.players_open_melds)
                players_discarded_tiles = deepcopy(self.players_discarded_tiles)
                can_peng, can_chi, can_gang, can_bu_gang, can_an_gang = False, False, False, False, False
                tiles_can_bu_gang, tiles_can_an_gang = [], []
                state4discard = None
                self_tiles = deepcopy(self.players_tiles[player_id])
                if player_id == action_player_id:
                    if isinstance(action, (Chi, Peng, Gang)):
                        # 取消上一个action时设置的默认玩家Pass其他玩家打出的牌
                        # 因为数据处理时，当一位玩家打出牌后，我首先默认的所有玩家Pass，如果有玩家吃碰杠，是在下一个action里处理的，会有标签冲突问题
                        if action_sequences[player_id][-1] == "Pass":
                            state_sequences[player_id].pop()
                            action_sequences[player_id].pop()
                    if isinstance(action, (Play, Chi, Peng, Gang, BuGang, AnGang)):
                        last_self_tiles4discard = deepcopy(self_tiles)
                        players_open_melds4discard = deepcopy(players_open_melds)
                        state4discard = {
                            "self_tiles": last_self_tiles4discard,
                            "players_open_melds": players_open_melds4discard
                        }
                        if isinstance(action, Play):
                            action.next_state_before_play_(last_self_tiles4discard)
                        else:
                            action.next_state_before_play_(
                                last_self_tiles4discard,
                                players_open_melds4discard[player_id]
                            )
                    if isinstance(action, Chi):
                        can_chi = True
                    if isinstance(action, Peng):
                        can_peng = True
                    if isinstance(action, Gang):
                        can_gang = True
                    if isinstance(action, BuGang):
                        can_bu_gang = True
                    if isinstance(action, AnGang):
                        can_an_gang = True
                    if not isinstance(action, (Chi, Peng)):
                        for self_tile in self_tiles:
                            if self_tiles[self_tile] >= 4:
                                can_an_gang = True
                                tiles_can_an_gang.append(self_tile)
                        open_melds = players_open_melds[action_player_id]
                        for open_meld_type, _, open_meld_tile in open_melds:
                            if open_meld_type == "Peng" and self_tiles[open_meld_tile] >= 1:
                                can_bu_gang = True
                                tiles_can_bu_gang.append(open_meld_tile)
                    action_sequences[player_id].append(action)
                else:
                    if (last_action_player_id is not None) or (action_i == 0):
                        if action_i == 0:
                            # todo: 没有考虑起手摸牌就和（国标没有天地人和），后面直接把这种牌局剔除了
                            last_action_player_id = 0
                            last_tile = action.tile_out
                        else:
                            last_tile = last_action.tile_out
                        if (last_tile < 27) and (player_id == (last_action_player_id - 1) or ((player_id == 0) and (last_action_player_id == 3))):
                            # 检查是否能吃
                            chi_tile_pairs = []
                            tile_color = last_tile // 9
                            tile2chi = last_tile % 9
                            # 三张牌的顺子必须是连续的，如 (tile-2, tile-1, tile)
                            # 所以我们只需要找出哪些组合以 tile 为结尾、中间或开头
                            for offset in [-2, -1, 0]:
                                a = tile2chi + offset
                                b = tile2chi + offset + 1
                                c = tile2chi + offset + 2
                                if 0 <= a <= 8 and 0 <= b <= 8 and 0 <= c <= 8:
                                    if tile2chi == a:
                                        chi_tile_pairs.append((tile_color * 9 + b, tile_color * 9 + c))
                                    elif tile2chi == b:
                                        chi_tile_pairs.append((tile_color * 9 + a, tile_color * 9 + c))
                                    elif tile2chi == c:
                                        chi_tile_pairs.append((tile_color * 9 + a, tile_color * 9 + b))
                            for chi_tile_pair in chi_tile_pairs:
                                tile1, tile2 = chi_tile_pair
                                if self_tiles[tile1] > 0 and self_tiles[tile2] > 0:
                                    can_chi = True
                                    break
                        if self_tiles[last_tile] >= 2:
                            can_peng = True
                        if self_tiles[last_tile] >= 3:
                            can_gang = True
                    action_sequences[player_id].append("Pass")
                state_sequences[player_id].append({
                    "can_chi": can_chi,
                    "can_peng": can_peng,
                    "can_gang": can_gang,
                    "can_bu_gang": can_bu_gang,
                    "tiles_can_bu_gang": tiles_can_bu_gang,
                    "can_an_gang": can_an_gang,
                    "tiles_can_an_gang": tiles_can_an_gang,
                    "last_action_player_id": last_action_player_id,
                    "last_action": last_action,
                    "self_tiles": self_tiles,
                    "players_open_melds": players_open_melds,
                    "players_discarded_tiles": players_discarded_tiles,
                    "state4discard": state4discard
                })
        last_action_player_id, last_action = self.action_sequence[-1]
        self.update_state(last_action_player_id, last_action)
        return state_sequences, action_sequences

    @staticmethod
    def read_one_game(data_path):
        with open(data_path, "r") as f:
            data = f.read().strip()
            data_lines = data.split("\n")
        return GameData(data_lines)
