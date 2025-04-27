import numpy as np

from pymj.agent.chinese_official_mahjiong.Agent import Agent
from pymj.agent.chinese_official_mahjiong.utils import choose_card2chi


class FuLuRandomAgent(Agent):
    def __init__(self, random_generator: np.random.RandomState, chi: bool = True, peng: bool = True, gang: bool = True, use_choose_card2_chi: bool = True):
        self.random_generator: np.random.RandomState = random_generator
        self.chi: bool = chi
        self.peng: bool = peng
        self.gang: bool = gang
        self.use_choose_card2_chi: bool = use_choose_card2_chi

    def choose_play(self, self_hand_card_ids: list[int], state: dict) -> int:
        return self.random_generator.choice(self_hand_card_ids)

    def choose_chi(self, self_hand_card_ids: list[int], state: dict, middle_card_ids: list[int]) -> int:
        return (
            choose_card2chi(self_hand_card_ids, middle_card_ids) if \
                self.use_choose_card2_chi else self.random_generator.choice(middle_card_ids)
        ) if self.chi else -1

    def choose_peng(self, self_hand_card_ids: list[int], state: dict, card2peng_id: int) -> bool:
        return self.peng

    def choose_gang(self, self_hand_card_ids: list[int], state: dict, card2gang_id: int) -> bool:
        return self.gang

    def choose_bu_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        return self.random_generator.choice(gang_card_ids) if self.gang else -1

    def choose_an_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        return self.random_generator.choice(gang_card_ids) if self.gang else -1

    def choose_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True

    def choose_zi_mo_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True
