import torch
import numpy as np

from pymj.agent.chinese_official_mahjiong.Agent import Agent
from pymj.agent.chinese_official_mahjiong.DQNAgent import Network, DeepNetwork


class SLBasedAgent(Agent):
    def __init__(self, random_generator: np.random.RandomState, play_model_path: str = None, chi_model_path: str = None,
                 peng_model_path: str = None, gang_model_path: str = None, device: str = "cpu", deep: bool = False, 
                 n1: float = 0.55, n2: float = 1.0):
        self.random_generator: np.random.RandomState = random_generator
        self.device: str = device
        self.n1: float = n1
        self.n2: float = n2
        if deep:
            self.play_model = SLBasedAgent.load_deep_model(play_model_path, "Play", self.device)
            self.chi_model = SLBasedAgent.load_deep_model(chi_model_path, "Chi", self.device)
            self.peng_model = SLBasedAgent.load_deep_model(peng_model_path, "Peng", self.device)
            self.gang_model = SLBasedAgent.load_deep_model(gang_model_path, "Gang", self.device)
        else:
            self.play_model = SLBasedAgent.load_model(play_model_path, "Play", self.device)
            self.chi_model = SLBasedAgent.load_model(chi_model_path, "Chi", self.device)
            self.peng_model = SLBasedAgent.load_model(peng_model_path, "Peng", self.device)
            self.gang_model = SLBasedAgent.load_model(gang_model_path, "Gang", self.device)
        self.peng_model.eval()
        self.chi_model.eval()
        self.play_model.eval()
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
    def load_deep_model(model_path: str, model_type: str, device: str):
        assert model_type in ["Play", "Chi", "Peng", "Gang"]
        if model_type == "Play":
            model = DeepNetwork(34, 288, 64)
        elif model_type == "Gang":
            model = DeepNetwork(4, 288, 64)
        else:
            model = DeepNetwork(1, 288, 64)
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")["state_dict"]
        )
        return model.to(device)

    def get_features(self, self_hand_card_ids: list[int], state: dict) -> tuple[torch.Tensor, ...]:
        self_wind = int(state["self_wind"])
        self_player_id = self_wind
        game_wind = int(state["game_wind"])
        players_melds = state["players_melds"]
        players_played_card_ids = state["players_played_card_ids"]
        wind_features = Network.feature_wind(self_wind, game_wind)
        self_hand_card_features4mlp, features4cnn = Network.feature_self_cards(
            self_hand_card_ids)
        self_played_card_features4mlp, self_played_card_features4rnn, self_seq_len = (
            Network.feature_played_cards(players_played_card_ids[self_player_id], 21))
        self_melds_features = Network.feature_melds(players_melds[self_player_id])
        other_played_card_ids = []
        for i in range(4):
            if i != self_player_id:
                other_played_card_ids.extend(players_played_card_ids[i])
        other_played_card_features4mlp, _, _ = Network.feature_played_cards(
            other_played_card_ids, 21
        )
        left_player_id = (self_player_id - 1) if (self_player_id > 0) else 3
        right_player_id = (self_player_id + 1) if (self_player_id < 3) else 0
        across_player_id = (self_player_id + 2) if (self_player_id < 2) else (self_player_id - 2)
        _, left_played_card_features4rnn, left_seq_len = (
            Network.feature_played_cards(players_played_card_ids[left_player_id], 21))
        _, right_played_card_features4rnn, right_seq_len = (
            Network.feature_played_cards(players_played_card_ids[right_player_id], 21))
        _, across_played_card_features4rnn, across_seq_len = (
            Network.feature_played_cards(players_played_card_ids[across_player_id], 21))
        remain_card_features = Network.feature_remain_cards(
            self_player_id, self_hand_card_ids, tuple(players_played_card_ids), players_melds)
        left_melds_features = Network.feature_melds(players_melds[left_player_id])
        right_melds_features = Network.feature_melds(players_melds[right_player_id])
        across_melds_features = Network.feature_melds(players_melds[across_player_id])
        features4mlp = torch.cat(
            (wind_features, self_hand_card_features4mlp, self_played_card_features4mlp,
                self_melds_features, other_played_card_features4mlp, left_melds_features,
                right_melds_features, across_melds_features, remain_card_features), dim=0
        ).unsqueeze(dim=0)
        features4rnn = torch.cat((
            self_played_card_features4rnn.unsqueeze(dim=0),
            left_played_card_features4rnn.unsqueeze(dim=0),
            right_played_card_features4rnn.unsqueeze(dim=0),
            across_played_card_features4rnn.unsqueeze(dim=0)
        ), dim=0).unsqueeze(dim=0)
        features4cnn = features4cnn.unsqueeze(dim=0)
        rnn_seqs_len = torch.cat([
            self_seq_len.unsqueeze(dim=0), 
            left_seq_len.unsqueeze(dim=0), 
            right_seq_len.unsqueeze(dim=0), 
            across_seq_len.unsqueeze(dim=0)], dim=0)
        return features4mlp.to(self.device), features4cnn.to(self.device), features4rnn.to(self.device), rnn_seqs_len.to(self.device)

    def choose_play(self, self_hand_card_ids: list[int], state: dict) -> int:
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = self.get_features(self_hand_card_ids, state)
        model_output: torch.Tensor = self.play_model(features4mlp, features4cnn, features4rnn, rnn_seqs_len)
        card2play_ids: list[int] = torch.sort(model_output[0], descending=True)[1].tolist()
        i: int = 0
        card2play_id: int = card2play_ids[i]
        while card2play_id not in self_hand_card_ids:
            i += 1
            card2play_id = card2play_ids[i]
        return card2play_id

    def choose_chi(self, self_hand_card_ids: list[int], state: dict, middle_card_ids: list[int]) -> int:
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = self.get_features(self_hand_card_ids, state)
        do_chi = self.chi_model(features4mlp, features4cnn, features4rnn, rnn_seqs_len).squeeze(dim=-1).detach().cpu().item() > self.n1
        if do_chi:
            return self.random_generator.choice(middle_card_ids)
        else:
            return -1

    def choose_peng(self, self_hand_card_ids: list[int], state: dict, card2peng_id: int) -> bool:
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = self.get_features(self_hand_card_ids, state)
        return self.peng_model(features4mlp, features4cnn, features4rnn, rnn_seqs_len).squeeze(dim=-1).detach().cpu().item() > self.n1

    def choose_gang(self, self_hand_card_ids: list[int], state: dict, card2gang_id: int) -> bool:
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = self.get_features(self_hand_card_ids, state)
        model_output = self.gang_model(features4mlp, features4cnn, features4rnn, rnn_seqs_len)
        return (model_output[0][2] / model_output[0][3]) >= self.n2
    
    def choose_bu_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = self.get_features(self_hand_card_ids, state)
        model_output = self.gang_model(features4mlp, features4cnn, features4rnn, rnn_seqs_len)
        do_gang = (model_output[0][1] / model_output[0][3]) >= self.n2
        if do_gang:
            return self.random_generator.choice(gang_card_ids)
        else:
            return -1

    def choose_an_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = self.get_features(self_hand_card_ids, state)
        model_output = self.gang_model(features4mlp, features4cnn, features4rnn, rnn_seqs_len)
        do_gang = (model_output[0][0] / model_output[0][3]) >= self.n2
        if do_gang:
            return self.random_generator.choice(gang_card_ids)
        else:
            return -1

    def choose_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True

    def choose_zi_mo_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True
