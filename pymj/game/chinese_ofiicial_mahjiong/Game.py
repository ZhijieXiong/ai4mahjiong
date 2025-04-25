import numpy as np
from copy import deepcopy

from pymj.agent.chinese_official_mahjiong.Agent import Agent
from pymj.game.chinese_ofiicial_mahjiong.Card import Card
from pymj.game.chinese_ofiicial_mahjiong.Player import Player


class Game:
    def __init__(self, agents: list[Agent], game_wind: int, players_self_wind: list[int], random_generator: np.random.RandomState = None, random_seed: int = 0):
        assert len(agents) == 4
        self.agents: list[Agent] = agents
        self.players: list[Player] = []
        self.game_wind: int = game_wind
        self.players_self_wind: list[int] = players_self_wind
        if random_generator is None:
            self.random_generator: np.random.RandomState = np.random.RandomState(random_seed)
        else:
            self.random_generator: np.random.RandomState = random_generator
        self.card_walls: list[list[int]] = []
        self.history: list[tuple[int, str, int]] = []
        self.last_observation: tuple[int, str, int] = (0, "", -1)
        self.players_played_card_actions: tuple[list[str], ...] = tuple([[] for _ in range(4)])
        self.players_played_card_ids: tuple[list[int], ...] = tuple([[] for _ in range(4)])
        self.players_melds: tuple[list[tuple[str, int, int]], ...] = tuple([[] for _ in range(4)])
        self.initial_players_card_ids: list[list[int]] = []
        self.game_result: tuple[list[int], list] = ([], [])

    def get_state(self) -> dict:
        return {
            "game_wind": self.game_wind,
            "players_played_card_ids": self.players_played_card_ids,
            "players_melds": self.players_melds,
            "last_observation": self.last_observation,
            "is_last_card": sum(list(map(len, self.card_walls))) == 0
        }

    def init_game(self, initial_card_walls: list[list[int]]) -> None:
        for player_id, agent in enumerate(self.agents):
            self.players.append(Player(player_id, agent, self.players_self_wind[player_id]))
        self.card_walls = deepcopy(initial_card_walls)
        self.deal_initial_cards()
        for i in range(4):
            self.initial_players_card_ids.append(deepcopy(self.players[i].hand_card_ids))

    def reset_game(self,  initial_card_walls: list[list[int]], game_wind: int, players_self_wind: list[int]) -> None:
        self.players = []
        self.game_wind = game_wind
        self.players_self_wind = players_self_wind
        self.card_walls = []
        self.history = []
        self.last_observation = (0, "", -1)
        self.players_played_card_actions = tuple([[] for _ in range(4)])
        self.players_played_card_ids = tuple([[] for _ in range(4)])
        self.players_melds = tuple([[] for _ in range(4)])
        self.initial_players_card_ids = []
        self.game_result = ([], [])
        self.init_game(initial_card_walls)

    def run(self):
        while not self.game_over():
            # print(f"[DEBUG] Last observation: {self.last_observation}")
            if self.last_observation[1] in ["Draw", "Chi", "Peng"]:
                # for i in range(4):
                #     num_hand_card = len(self.players[i].hand_card_ids)
                #     num_meld = len(self.players_melds[i])
                #     if i == self.last_observation[0]:
                #         assert num_hand_card == (14 - 3 * num_meld), f"玩家{i}副露数为{num_meld}，手牌数为{num_hand_card}"
                #     else:
                #         assert num_hand_card == (13 - 3 * num_meld), f"玩家{i}副露数为{num_meld}，手牌数为{num_hand_card}"
                self.play_loop()
            else:
                # for i in range(4):
                #     num_hand_card = len(self.players[i].hand_card_ids)
                #     num_meld = len(self.players_melds[i])
                #     assert num_hand_card == (13 - 3 * num_meld), f"玩家{i}副露数为{num_meld}，手牌数为{num_hand_card}"
                self.draw_loop()

    def play_loop(self):
        self.request_play()
        action_options: dict = {
            "Chi": (-1, []),
            "Peng": (-1, -1),
            "Gang": (-1, -1),
            "Hu": []
        }
        # 判断其它玩家能否胡、吃、碰、杠
        for i in range(4):
            if i == self.last_observation[0]:
                continue
            target_player: Player = self.players[i]
            fan_result: list[tuple[int, str]] = target_player.check_hu(self.get_state())
            if len(fan_result) > 0:
                action_options["Hu"].append((i, fan_result))
            card2gang_id: int = target_player.check_gang(self.get_state())
            if card2gang_id != -1:
                action_options["Gang"] = (i, card2gang_id)
            card2peng_id: int = target_player.check_peng(self.get_state())
            if card2peng_id != -1:
                action_options["Peng"] = (i, card2peng_id)
            if i == (self.last_observation[0] + 1) % 4:
                middle_card_ids = target_player.check_chi(self.get_state())
                if len(middle_card_ids) > 0:
                    action_options["Chi"] = (i, middle_card_ids)
        someone_hu: bool = False
        if len(action_options["Hu"]) > 0:
            for i, fan_result in action_options["Hu"]:
                target_player: Player = self.players[i]
                if target_player.response_hu(self.get_state(), fan_result):
                    self.game_result[0].append(i)
                    self.game_result[1].append(fan_result)
                    someone_hu = True
        if someone_hu:
            return
        if action_options["Gang"][0] != -1:
            gang_player_id: int = action_options["Gang"][0]
            card2gang_id: int = action_options["Gang"][1]
            self.request_gang(gang_player_id, card2gang_id)
        elif action_options["Peng"][0] != -1:
            peng_player_id: int = action_options["Peng"][0]
            card2peng_id: int = action_options["Peng"][1]
            self.request_peng(peng_player_id, card2peng_id)
        elif action_options["Chi"][0] != -1:
            chi_player_id: int = action_options["Chi"][0]
            middle_card_ids: list[int] = action_options["Chi"][1]
            self.request_chi(chi_player_id, middle_card_ids)

    def draw_loop(self):
        if not self.request_draw():
            return
        target_player: Player = self.players[self.last_observation[0]]
        fan_result: list[tuple[int, str]] = target_player.check_zi_mo_hu(self.get_state())
        if len(fan_result) > 0:
            if target_player.response_hu(self.get_state(), fan_result):
                self.game_result[0].append(target_player.get_id())
                self.game_result[1].append(fan_result)
                return
        # 如果不能胡，但已经是最后一张牌了，也应该打一张牌
        if sum(list(map(len, self.card_walls))) == 0:
            self.request_play()
            return
        can_bu_gang = target_player.check_bu_gang(self.get_state())
        if len(can_bu_gang) > 0:
            # todo: 需要考虑抢杠胡
            pass
        can_an_gang = target_player.check_an_gang()
        if len(can_an_gang) > 0:
            # todo:
            pass

    @staticmethod
    def get_initial_card_walls(random_generator):
        card_walls = []
        all_card_ids: list[int] = list(range(34)) * 4
        random_generator.shuffle(all_card_ids)
        for i in range(4):
            card_walls.append(all_card_ids[34 * i:34 * (i + 1)])
        return card_walls

    def deal_initial_cards(self):
        for i in range(4):
            player = self.players[i]
            player_id = player.get_id()
            for _ in range(13):
                player.response_draw(self.card_walls[player_id].pop())

    def request_draw(self) -> bool:
        last_player_id: int = self.last_observation[0]
        last_action: str = self.last_observation[1]
        if last_action == "Play":
            current_player_index = (last_player_id + 1) % 4
        else:
            current_player_index = last_player_id
        # 避免一开始直接取错
        next_draw_index = current_player_index
        # 寻找有牌的牌墙
        for _ in range(4):  # 最多尝试4次
            card_wall = self.card_walls[next_draw_index]
            if len(card_wall) > 0:
                break
            next_draw_index = (next_draw_index + 1) % 4
        else:
            return False  # 没有可摸的牌了
        # 真正要摸牌的玩家
        current_player = self.players[next_draw_index]
        card2draw_id = self.card_walls[next_draw_index].pop()
        current_player.response_draw(card2draw_id)
        self.last_observation = (next_draw_index, "Draw", card2draw_id)
        self.history.append(deepcopy(self.last_observation))
        return True

    def request_play(self) -> None:
        current_player: Player = self.players[self.last_observation[0]]
        card2play_id: int = current_player.response_play(self.get_state())
        self.players_played_card_actions[current_player.get_id()].append(self.last_observation[1])
        self.players_played_card_ids[current_player.get_id()].append(card2play_id)
        self.last_observation = (current_player.get_id(), "Play", card2play_id)
        self.history.append(deepcopy(self.last_observation))

    def request_chi(self, player_id: int, middle_card_ids: list[int]) -> None:
        middle_card_id: int = self.players[player_id].response_chi(self.get_state(), middle_card_ids)
        if middle_card_id != -1:
            last_player_id: int = self.last_observation[0]
            self.pop_last_player_played_card(last_player_id)
            self.change_played_card(player_id, "Chi", last_player_id, middle_card_id)
            self.request_play()

    def request_peng(self, player_id: int, card2peng_id: int) -> None:
        if self.players[player_id].response_peng(self.get_state(), card2peng_id):
            last_player_id: int = self.last_observation[0]
            self.pop_last_player_played_card(last_player_id)
            self.change_played_card(player_id, "Peng", last_player_id, card2peng_id)
            self.request_play()

    def request_gang(self, player_id: int, card2gang_id: int) -> None:
        if self.players[player_id].response_gang(self.get_state(), card2gang_id):
            last_player_id: int = self.last_observation[0]
            self.pop_last_player_played_card(last_player_id)
            self.change_played_card(player_id, "Gang", last_player_id, card2gang_id)
            self.request_draw()
            self.request_play()

    def request_bu_gang(self, player_id: int, gang_card_ids: list[int]) -> None:
        card2gang_id: int = self.players[player_id].response_an_gang(self.get_state(), gang_card_ids)
        if card2gang_id != -1:
            for i, (meld_type, _, meld_card_id) in enumerate(self.players_melds[player_id]):
                if meld_type == "Peng" and meld_card_id == card2gang_id:
                    self.players_melds[player_id][i] = ("BuGang", player_id, card2gang_id)
                    break
            self.last_observation = (player_id, "BuGang", card2gang_id)
            self.request_draw()
            self.request_play()

    def request_an_gang(self, player_id: int, gang_card_ids: list[int]) -> None:
        card2gang_id: int = self.players[player_id].response_an_gang(self.get_state(), gang_card_ids)
        if card2gang_id != -1:
            self.players_melds[player_id].append(("AnGang", player_id, card2gang_id))
            self.last_observation = (player_id, "AnGang", card2gang_id)
            self.request_draw()
            self.request_play()

    def change_played_card(self, player_id, action_type, last_player_id, card_id):
        self.players_melds[player_id].append((action_type, last_player_id, card_id))
        self.last_observation = (player_id, action_type, card_id)
        self.history.append(deepcopy(self.last_observation))

    def pop_last_player_played_card(self, last_player_id: int) -> None:
        self.players_played_card_actions[last_player_id].pop()
        self.players_played_card_ids[last_player_id].pop()

    def game_over(self) -> bool:
        num_remain_card = sum(list(map(len, self.card_walls)))
        return (num_remain_card == 0) or (len(self.game_result[0]) > 0)

    def print_game(self) -> None:
        print(f"Wind {self.game_wind}")
        for i in range(4):
            hand_cards_str: str = " ".join(list(map(Card.decoding, self.initial_players_card_ids[i])))
            print(f"Player {i} Deal " + hand_cards_str)
        for player_id, action_type, card_id in self.history:
            print(f"Player {player_id} {action_type} {Card.decoding(card_id)}")
        if len(self.game_result[0]) == 0:
            print("Huang")
        elif len(self.game_result[0]) == 1:
            hu_player_id: int = self.game_result[0][0]
            last_card_id: int = self.last_observation[2]
            print(f"Player {hu_player_id} Hu {Card.decoding(last_card_id)}")
            print(self.game_result[1][0])
        else:
            last_player_id: int = self.last_observation[0]
            last_card_id: int = self.last_observation[2]
            for i in range(1, 4):
                nearest_player_id: int = (last_player_id + i) % 4
                if nearest_player_id in self.game_result[0]:
                    idx: int = self.game_result[0].index(nearest_player_id)
                    print(f"Player {nearest_player_id} Hu {Card.decoding(last_card_id)}")
                    print(self.game_result[1][idx])
