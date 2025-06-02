from copy import deepcopy

from pymj.botzone.util import *


class Play:
    def __init__(self, tile_str_in, tile_str_out):
        self.tile_in = tile_str2int(tile_str_in)
        self.tile_out = tile_str2int(tile_str_out)

    def next_tiles_(self, last_self_tiles, self_discarded_tiles):
        last_self_tiles[self.tile_in] += 1
        last_self_tiles[self.tile_out] -= 1
        self_discarded_tiles.append(("Play", self.tile_out))

    def next_state_before_play_(self, last_self_tiles):
        last_self_tiles[self.tile_in] += 1


class Chi:
    def __init__(self, played_player_id, tile_str_in, tile_str_chi_middle, tile_str_out):
        self.played_player_id = played_player_id
        self.tile_in = tile_str2int(tile_str_in)
        self.tile_chi_middle = tile_str2int(tile_str_chi_middle)
        self.tile_out = tile_str2int(tile_str_out)

    def next_tiles_(self, last_self_tiles, last_self_open_melds, other_discarded_tiles, self_discarded_tiles):
        if self.tile_in < self.tile_chi_middle:
            tile_chi_1 = self.tile_chi_middle
            tile_chi_2 = self.tile_chi_middle + 1
        elif self.tile_in > self.tile_chi_middle:
            tile_chi_1 = self.tile_chi_middle
            tile_chi_2 = self.tile_chi_middle - 1
        else:
            tile_chi_1 = self.tile_chi_middle - 1
            tile_chi_2 = self.tile_chi_middle + 1

        last_self_tiles[tile_chi_1] -= 1
        last_self_tiles[tile_chi_2] -= 1
        last_self_tiles[self.tile_out] -= 1
        last_self_open_melds.append(("Chi", self.played_player_id, self.tile_chi_middle))
        other_discarded_tiles.pop()
        self_discarded_tiles.append(("Chi", self.tile_out))

    def next_state_before_play_(self, last_self_tiles, last_self_open_melds):
        if self.tile_in < self.tile_chi_middle:
            tile_chi_1 = self.tile_chi_middle
            tile_chi_2 = self.tile_chi_middle + 1
        elif self.tile_in > self.tile_chi_middle:
            tile_chi_1 = self.tile_chi_middle
            tile_chi_2 = self.tile_chi_middle - 1
        else:
            tile_chi_1 = self.tile_chi_middle - 1
            tile_chi_2 = self.tile_chi_middle + 1

        last_self_tiles[tile_chi_1] -= 1
        last_self_tiles[tile_chi_2] -= 1
        last_self_open_melds.append(("Chi", self.played_player_id, self.tile_chi_middle))


class Peng:
    def __init__(self, played_player_id, tile_str_in, tile_str_out):
        self.played_player_id = played_player_id
        self.tile_in = tile_str2int(tile_str_in)
        self.tile_out = tile_str2int(tile_str_out)

    def next_tiles_(self, last_self_tiles, last_self_open_melds, other_discarded_tiles, self_discarded_tiles):
        last_self_tiles[self.tile_in] -= 2
        last_self_tiles[self.tile_out] -= 1
        last_self_open_melds.append(("Peng", self.played_player_id, self.tile_in))
        other_discarded_tiles.pop()
        self_discarded_tiles.append(("Peng", self.tile_out))

    def next_state_before_play_(self, last_self_tiles, last_self_open_melds):
        last_self_tiles[self.tile_in] -= 2
        last_self_open_melds.append(("Peng", self.played_player_id, self.tile_in))


class Gang:
    def __init__(self, played_player_id, tile_str_gang, tile_str_in, tile_str_out):
        self.played_player_id = played_player_id
        self.tile_gang = tile_str2int(tile_str_gang)
        self.tile_in = tile_str2int(tile_str_in)
        self.tile_out = tile_str2int(tile_str_out)

    def next_tiles_(self, last_self_tiles, last_self_open_melds, other_discarded_tiles, self_discarded_tiles):
        last_self_tiles[self.tile_gang] -= 3
        last_self_tiles[self.tile_in] += 1
        last_self_tiles[self.tile_out] -= 1
        last_self_open_melds.append(("Gang", self.played_player_id, self.tile_gang))
        other_discarded_tiles.pop()
        self_discarded_tiles.append(("Gang", self.tile_out))

    def next_state_before_play_(self, last_self_tiles, last_self_open_melds):
        last_self_tiles[self.tile_gang] -= 3
        last_self_tiles[self.tile_in] += 1
        last_self_open_melds.append(("Gang", self.played_player_id, self.tile_gang))


class AnGang:
    def __init__(self, tile_str_in1, tile_str_gang, tile_str_in2, tile_str_out):
        # 摸，杠，摸，打
        self.tile_in1 = tile_str2int(tile_str_in1)
        self.tile_gang = tile_str2int(tile_str_gang)
        self.tile_in2 = tile_str2int(tile_str_in2)
        self.tile_out = tile_str2int(tile_str_out)

    def next_tiles_(self, last_self_tiles, last_self_open_melds, self_discarded_tiles):
        last_self_tiles[self.tile_in1] += 1
        last_self_tiles[self.tile_gang] -= 4
        last_self_tiles[self.tile_in2] += 1
        last_self_tiles[self.tile_out] -= 1
        last_self_open_melds.append(("AnGang", -1, self.tile_gang))
        self_discarded_tiles.append(("AnGang", self.tile_out))

    def next_state_before_play_(self, last_self_tiles, last_self_open_melds):
        last_self_tiles[self.tile_in1] += 1
        last_self_tiles[self.tile_gang] -= 4
        last_self_tiles[self.tile_in2] += 1
        last_self_open_melds.append(("AnGang", -1, self.tile_gang))


class BuGang:
    def __init__(self, tile_str_in1, tile_str_gang, tile_str_in2, tile_str_out):
        # 摸，杠，摸，打
        self.tile_in1 = tile_str2int(tile_str_in1)
        self.tile_gang = tile_str2int(tile_str_gang)
        self.tile_in2 = tile_str2int(tile_str_in2)
        self.tile_out = tile_str2int(tile_str_out)

    def next_tiles_(self, last_self_tiles, last_self_open_melds, self_discarded_tiles):
        last_self_tiles[self.tile_in1] += 1
        last_self_tiles[self.tile_gang] -= 1
        last_self_tiles[self.tile_in2] += 1
        last_self_tiles[self.tile_out] -= 1
        open_meld_idx = 0
        for i, (open_meld_type, _, open_meld_tile) in enumerate(last_self_open_melds):
            if open_meld_type == "Peng" and open_meld_tile == self.tile_gang:
                open_meld_idx = i
                break
        last_self_open_melds[open_meld_idx] = ("BuGang", -1, self.tile_gang)
        self_discarded_tiles.append(("BuGang", self.tile_out))

    def next_state_before_play_(self, last_self_tiles, last_self_open_melds):
        last_self_tiles[self.tile_in1] += 1
        last_self_tiles[self.tile_gang] -= 1
        last_self_tiles[self.tile_in2] += 1
        open_meld_idx = 0
        for i, (open_meld_type, _, open_meld_tile) in enumerate(last_self_open_melds):
            if open_meld_type == "Peng" and open_meld_tile == self.tile_gang:
                open_meld_idx = i
                break
        last_self_open_melds[open_meld_idx] = ("BuGang", -1, self.tile_gang)


class Hu:
    def __init__(self, played_player_id, tile_str_in, fans_str):
        self.played_player_id = played_player_id
        self.tile_in = tile_str2int(tile_str_in)
        self.fans = fans_str2dict(fans_str)

    def next_tiles_(self, last_self_tiles):
        last_self_tiles[self.tile_in] += 1

    def tiles_before_play(self, last_self_tiles):
        tiles = deepcopy(last_self_tiles)
        tiles[self.tile_in] += 1
        return tiles


class ZiMoHu:
    def __init__(self, tile_str_in, fans_str):
        self.tile_in = tile_str2int(tile_str_in)
        self.fans = fans_str2dict(fans_str)

    def next_tiles_(self, last_self_tiles):
        last_self_tiles[self.tile_in] += 1

    def tiles_before_play(self, last_self_tiles):
        tiles = deepcopy(last_self_tiles)
        tiles[self.tile_in] += 1
        return tiles
