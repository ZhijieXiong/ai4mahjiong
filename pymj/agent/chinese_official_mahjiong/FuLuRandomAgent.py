import numpy as np

from pymj.agent.chinese_official_mahjiong.Agent import Agent


class FuLuRandomAgent(Agent):
    def __init__(self, random_generator: np.random.RandomState, chi: bool = True, peng: bool = True, gang: bool = True):
        self.random_generator: np.random.RandomState = random_generator
        self.chi = chi
        self.peng = peng
        self.gang = gang

    def choose_play(self, self_hand_card_ids: list[int], state: dict) -> int:
        return self.random_generator.choice(self_hand_card_ids)

    def choose_chi(self, self_hand_card_ids: list[int], state: dict, middle_card_ids: list[int]) -> int:
        return middle_card_ids[0] if self.chi else -1

    def choose_peng(self, self_hand_card_ids: list[int], state: dict, card2peng_id: int) -> bool:
        return self.peng

    def choose_gang(self, self_hand_card_ids: list[int], state: dict, card2gang_id: int) -> bool:
        return self.gang

    def choose_bu_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        return gang_card_ids[0] if self.gang else -1

    def choose_an_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        return gang_card_ids[0] if self.gang else -1

    def choose_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True

    def choose_zi_mo_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True
