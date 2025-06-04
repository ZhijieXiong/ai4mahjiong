import torch
import numpy as np

from pymj.agent.chinese_official_mahjiong.Agent import Agent
from pymj.agent.chinese_official_mahjiong.DQNAgent import Network


class SLBasedAgent(Agent):
    def __init__(self, random_generator: np.random.RandomState, play_model_path: str = None, chi_model_path: str = None,
                 peng_model_path: str = None, gang_model_path: str = None, device: str = "cpu"):
        self.random_generator: np.random.RandomState = random_generator
        self.device: str = device
        self.play_model = SLBasedAgent.load_model(play_model_path, "Play", self.device)
        self.play_model.eval()
        self.chi_model = SLBasedAgent.load_model(chi_model_path, "Chi", self.device)
        self.chi_model.eval()
        self.peng_model = SLBasedAgent.load_model(peng_model_path, "Peng", self.device)
        self.peng_model.eval()
        self.gang_model = SLBasedAgent.load_model(gang_model_path, "Gang", self.device)
        self.gang_model.eval()

    @staticmethod
    def load_model(model_path: str, model_type: str, device: str):
        assert model_type in ["Play", "Chi", "Peng", "Gang"]
        if model_type == "Play":
            model = Network(34, 288, 64)
        elif model_type == "Gang":
            model = Network(4, 288, 64)
        else:
            model = Network(1, 288, 64)
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")["state_dict"]
        )
        return model.to(device)

    @staticmethod
    def get_features(state: dict) -> tuple[torch.Tensor, ...]:
        pass

    def choose_play(self, self_hand_card_ids: list[int], state: dict) -> int:
        all_features: tuple[torch.Tensor, ...] = self.get_features(state)
        model_output: torch.Tensor = self.play_model(features)
        card2play_ids: list[int] = torch.sort(model_output[0, :-1], descending=True)[1].tolist()
        i: int = 0
        card2play_id: int = card2play_ids[i]
        while card2play_id not in self_hand_card_ids:
            i += 1
            card2play_id = card2play_ids[i]
        return card2play_id

    def choose_chi(self, self_hand_card_ids: list[int], state: dict, middle_card_ids: list[int]) -> int:
        all_features: tuple[torch.Tensor, ...] = self.get_features(state)
        model_output: torch.Tensor = self.play_model(features)
        do_chi = self.chi_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5
        if do_chi:
            return self.random_generator.choice(middle_card_ids)
        else:
            return -1

    def choose_peng(self, self_hand_card_ids: list[int], state: dict, card2peng_id: int) -> bool:
        all_features: tuple[torch.Tensor, ...] = self.get_features(state)
        model_output: torch.Tensor = self.play_model(features)
        return self.peng_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5

    def choose_gang(self, self_hand_card_ids: list[int], state: dict, card2gang_id: int) -> bool:
        all_features: tuple[torch.Tensor, ...] = self.get_features(state)
        model_output: torch.Tensor = self.play_model(features)
        return self.gang_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5

    def choose_bu_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        all_features: tuple[torch.Tensor, ...] = self.get_features(state)
        model_output: torch.Tensor = self.play_model(features)
        do_gang = self.bu_gang_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5
        if do_gang:
            return self.random_generator.choice(gang_card_ids)
        else:
            return -1

    def choose_an_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        all_features: tuple[torch.Tensor, ...] = self.get_features(state)
        model_output: torch.Tensor = self.play_model(features)
        do_gang = self.an_gang_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5
        if do_gang:
            return self.random_generator.choice(gang_card_ids)
        else:
            return -1

    def choose_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True

    def choose_zi_mo_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True
