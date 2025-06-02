from collections import Counter
from copy import deepcopy
from MahjongGB import MahjongFanCalculator

from pymj.game.chinese_ofiicial_mahjiong.Card import Card
from pymj.agent.chinese_official_mahjiong.Agent import Agent


class Player:
    def __init__(self, player_id: int, agent: Agent, self_wind: int):
        self.id: int = player_id
        self.hand_card_ids: list[int] = []
        self.agent: Agent = agent
        self.self_wind: int = self_wind

    def get_id(self) -> int:
        return self.id

    def check_zi_mo_hu(self, state: dict) -> list[tuple[int, str]]:
        players_melds: list[tuple] = state["players_melds"]
        last_observation: tuple = state["last_observation"]
        last_action_card_id: int = last_observation[2]
        players_played_card_ids: tuple[list] = state["players_played_card_ids"]
        is_last_card: bool = state["is_last_card"]
        game_wind: int = state["game_wind"]

        # MahjongFanCalculator需要的数据
        pack = []
        for meld_type, meld_player_id, meld_card_id in players_melds[self.id]:
            if meld_type == "Chi":
                claiming = (meld_type.upper(), Card.decoding(meld_card_id), 1)
            else:
                claiming = (meld_type.replace("Bu", "").replace("An", "").upper(),
                            Card.decoding(meld_card_id),
                            (self.id - meld_player_id + 4) % 4)
            pack.append(claiming)
        # hand不应该计算摸上来那张牌
        self_hand_card_ids = deepcopy(self.hand_card_ids)
        self_hand_card_ids.remove(last_action_card_id)
        hand: tuple[str, ...] = tuple(map(Card.decoding, self_hand_card_ids))
        winTile: str = Card.decoding(last_action_card_id)
        flowerCount: int = 0
        isSelfDrawn: bool = True
        played_card_ids: dict = Counter([card_id for player_played_card_ids in players_played_card_ids for card_id in player_played_card_ids])
        for i in range(4):
            for meld_type, _, meld_card_id in players_melds[i]:
                if meld_type == "Chi":
                    played_card_ids[meld_card_id-1] += 1
                    played_card_ids[meld_card_id] += 1
                    played_card_ids[meld_card_id+1] += 1
                elif meld_type == "Peng":
                    played_card_ids[meld_card_id] += 3
                else:
                    played_card_ids[meld_card_id] += 4
        is4thTile: bool = played_card_ids[last_action_card_id] == 3
        isAboutKong: bool = False
        isWallLast: bool = is_last_card
        seatWind: int = self.self_wind
        prevalentWind: int = game_wind
        verbose: bool = False

        try:
            result = MahjongFanCalculator(tuple(pack), hand, winTile, flowerCount, isSelfDrawn, is4thTile,
                                          isAboutKong, isWallLast, seatWind, prevalentWind, verbose)
            fan_count: int = sum([res[0] for res in result])
            return result if fan_count >= 8 else []
        except:
            return []

    def check_hu(self, state: dict) -> list[tuple[int, str]]:
        players_melds: list[tuple] = state["players_melds"]
        last_action: str = state["last_observation"][1]
        last_action_card_id: int = state["last_observation"][2]
        players_played_card_ids: tuple[list] = state["players_played_card_ids"]
        is_last_card: bool = state["is_last_card"]
        game_wind: int = state["game_wind"]

        # MahjongFanCalculator需要的数据
        pack = []
        for meld_type, meld_player_id, meld_card_id in players_melds[self.id]:
            if meld_type == "Chi":
                claiming = (meld_type.upper(), Card.decoding(meld_card_id), 1)
            else:
                claiming = (meld_type.replace("Bu", "").replace("An", "").upper(),
                            Card.decoding(meld_card_id),
                            (self.id - meld_player_id + 4) % 4)
            pack.append(claiming)
        hand: tuple[str, ...] = tuple(map(Card.decoding, self.hand_card_ids))
        winTile: str = Card.decoding(last_action_card_id)
        flowerCount: int = 0
        isSelfDrawn: bool = False
        played_card_ids: dict = Counter([card_id for player_played_card_ids in players_played_card_ids for card_id in player_played_card_ids])
        for i in range(4):
            for meld_type, _, meld_card_id in players_melds[i]:
                if meld_type == "Chi":
                    played_card_ids[meld_card_id-1] += 1
                    played_card_ids[meld_card_id] += 1
                    played_card_ids[meld_card_id+1] += 1
                elif meld_type == "Peng":
                    played_card_ids[meld_card_id] += 3
                else:
                    played_card_ids[meld_card_id] += 4
        is4thTile: bool = played_card_ids[last_action_card_id] == 3
        isAboutKong: bool = last_action == "BuGang"
        isWallLast: bool = is_last_card
        seatWind: int = self.self_wind
        prevalentWind: int = game_wind
        verbose: bool = False

        try:
            result = MahjongFanCalculator(tuple(pack), hand, winTile, flowerCount, isSelfDrawn, is4thTile,
                                          isAboutKong, isWallLast, seatWind, prevalentWind, verbose)
            fan_count: int = sum([res[0] for res in result])
            return result if fan_count >= 8 else []
        except:
            return []

    def check_chi(self, state: dict) -> list[int]:
        hand_card_ids: list[int] = self.hand_card_ids
        card2chi_id: int = state["last_observation"][2]
        chi_options = []

        if card2chi_id >= 27:
            return chi_options

        # 三种可能的顺子组合：
        # card2chi_id - 2, card2chi_id - 1, card2chi_id
        # card2chi_id - 1, card2chi_id, card2chi_id + 1
        # card2chi_id, card2chi_id + 1, card2chi_id + 2
        # 需要判断花色一致性（每种花色编号区间跨度为9）
        base = (card2chi_id // 9) * 9
        # 顺子: card2chi_id - 2, card2chi_id - 1, card2chi_id
        if card2chi_id - 2 >= base:
            if (card2chi_id - 2 in hand_card_ids) and (card2chi_id - 1 in hand_card_ids):
                chi_options.append(card2chi_id - 1)
        # 顺子: card2chi_id - 1, card2chi_id, card2chi_id + 1
        if (card2chi_id - 1 >= base) and (card2chi_id + 1 < base + 9):
            if (card2chi_id - 1 in hand_card_ids) and (card2chi_id + 1 in hand_card_ids):
                chi_options.append(card2chi_id)
        # 顺子: card2chi_id, card2chi_id + 1, card2chi_id + 2
        if card2chi_id + 2 < base + 9:
            if (card2chi_id + 1 in hand_card_ids) and (card2chi_id + 2 in hand_card_ids):
                chi_options.append(card2chi_id + 1)

        return chi_options

    def check_peng(self, state: dict) -> int:
        card2peng_id: int = state["last_observation"][2]
        filtered_hand_cards = list(filter(lambda card_id: card_id == card2peng_id, self.hand_card_ids))
        return card2peng_id if len(filtered_hand_cards) >= 2 else -1

    def check_gang(self, state: dict) -> int:
        card2gang_id: int = state["last_observation"][2]
        filtered_hand_cards: list = list(filter(lambda card_id: card_id == card2gang_id, self.hand_card_ids))
        return card2gang_id if len(filtered_hand_cards) >= 3 else -1

    def check_bu_gang(self, state: dict) -> list:
        players_melds: list[tuple] = state["players_melds"]
        bu_gang_options = []

        for i, (meld_type, _, meld_card_id) in enumerate(players_melds[self.id]):
            if meld_type == "Peng" and meld_card_id in self.hand_card_ids:
                bu_gang_options.append(i)

        return bu_gang_options

    def check_an_gang(self) -> list:
        counter = Counter(self.hand_card_ids)
        return [elem for elem, count in counter.items() if count >= 4]

    def response_draw(self, card_id):
        self.hand_card_ids.append(card_id)
        self.hand_card_ids.sort()
        # self.log_hand_cards("摸牌后")

    def response_play(self, state: dict):
        state["player_id"] = self.id
        state["self_wind"] = self.self_wind
        card2play_id = self.agent.choose_play(self.hand_card_ids, state)
        assert card2play_id in self.hand_card_ids, "打出的牌不是自己的手牌"
        self.hand_card_ids.remove(card2play_id)
        # self.log_hand_cards("打牌后")
        return card2play_id

    def response_chi(self, state: dict, middle_card_ids: list[int]) -> int:
        state["player_id"] = self.id
        state["self_wind"] = self.self_wind
        middle_card_id: int = self.agent.choose_chi(self.hand_card_ids, state, middle_card_ids)
        if middle_card_id == -1:
            return -1
        else:
            card2chi_id: int = state["last_observation"][2]
            if card2chi_id < middle_card_id:
                card_id1 = middle_card_id
                card_id2 = middle_card_id + 1
            elif card2chi_id > middle_card_id:
                card_id1 = middle_card_id
                card_id2 = middle_card_id - 1
            else:
                card_id1 = middle_card_id - 1
                card_id2 = middle_card_id + 1
            assert card_id1 in self.hand_card_ids, f"没有牌{card_id1}，不能吃{card2chi_id}"
            self.hand_card_ids.remove(card_id1)
            assert card_id2 in self.hand_card_ids, f"没有牌{card_id2}，不能吃{card2chi_id}"
            self.hand_card_ids.remove(card_id2)
            # self.log_hand_cards("chi牌后")
            return middle_card_id

    def response_peng(self, state: dict, card2peng_id: int) -> bool:
        state["player_id"] = self.id
        state["self_wind"] = self.self_wind
        if self.agent.choose_peng(self.hand_card_ids, state, card2peng_id):
            assert card2peng_id in self.hand_card_ids, "没有准备杠的牌"
            self.hand_card_ids.remove(card2peng_id)
            assert card2peng_id in self.hand_card_ids, "准备碰的牌只有1张"
            self.hand_card_ids.remove(card2peng_id)
            # self.log_hand_cards("peng牌后")
            return True
        else:
            return False

    def response_gang(self, state: dict, card2gang_id: int) -> bool:
        state["player_id"] = self.id
        state["self_wind"] = self.self_wind
        if self.agent.choose_gang(self.hand_card_ids, state, card2gang_id):
            assert card2gang_id in self.hand_card_ids, "没有准备杠的牌"
            self.hand_card_ids.remove(card2gang_id)
            assert card2gang_id in self.hand_card_ids, "准备杠的牌只有1张"
            self.hand_card_ids.remove(card2gang_id)
            assert card2gang_id in self.hand_card_ids, "准备杠的牌只有2张"
            self.hand_card_ids.remove(card2gang_id)
            # self.log_hand_cards("gang牌后")
            return True
        else:
            return False

    def response_bu_gang(self, state: dict, gang_card_ids: list[int]):
        state["player_id"] = self.id
        state["self_wind"] = self.self_wind
        card2gang_id: int = self.agent.choose_bu_gang(self.hand_card_ids, state, gang_card_ids)
        if card2gang_id == -1:
            return -1
        else:
            assert card2gang_id in self.hand_card_ids, "没有准备杠的牌"
            self.hand_card_ids.remove(card2gang_id)
            return card2gang_id

    def response_an_gang(self, state: dict, gang_card_ids: list[int]):
        state["player_id"] = self.id
        state["self_wind"] = self.self_wind
        card2gang_id: int = self.agent.choose_an_gang(self.hand_card_ids, state, gang_card_ids)
        if card2gang_id == -1:
            return -1
        else:
            assert card2gang_id in self.hand_card_ids, "没有准备杠的牌"
            self.hand_card_ids.remove(card2gang_id)
            assert card2gang_id in self.hand_card_ids, "准备杠的牌只有1张"
            self.hand_card_ids.remove(card2gang_id)
            assert card2gang_id in self.hand_card_ids, "准备杠的牌只有2张"
            self.hand_card_ids.remove(card2gang_id)
            assert card2gang_id in self.hand_card_ids, "准备杠的牌只有3张"
            self.hand_card_ids.remove(card2gang_id)
            return card2gang_id

    def response_hu(self, state: dict, fan_result: list[tuple[int, str]]) -> bool:
        state["player_id"] = self.id
        state["self_wind"] = self.self_wind
        return self.agent.choose_hu(self.hand_card_ids, state, fan_result)

    def response_zi_mo_hu(self, state: dict, fan_result: list[tuple[int, str]]) -> bool:
        state["player_id"] = self.id
        state["self_wind"] = self.self_wind
        return self.agent.choose_zi_mo_hu(self.hand_card_ids, state, fan_result)

    def log_hand_cards(self, tag=""):
        print(f"[Player {self.id}] {tag} 手牌数量: {len(self.hand_card_ids)}, 手牌: {self.hand_card_ids}")
