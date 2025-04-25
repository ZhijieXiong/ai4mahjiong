from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def choose_play(self, self_hand_card_ids: list[int], state: dict) -> int:
        pass

    @abstractmethod
    def choose_chi(self, self_hand_card_ids: list[int], state: dict, middle_card_ids: list[int]) -> int:
        """middle_card_ids是可以吃的顺子中间张牌的id，-1表示不吃"""
        pass

    @abstractmethod
    def choose_peng(self, self_hand_card_ids: list[int], state: dict, card2peng_id: int) -> bool:
        pass

    @abstractmethod
    def choose_gang(self, self_hand_card_ids: list[int], state: dict, card2gang_id: int) -> bool:
        pass

    @abstractmethod
    def choose_bu_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        """gang_card_ids是可以杠的牌id，-1表示不吃"""
        pass

    @abstractmethod
    def choose_an_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        """gang_card_ids是可以杠的牌id，-1表示不吃"""
        pass

    @abstractmethod
    def choose_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True

    @abstractmethod
    def choose_zi_mo_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True
