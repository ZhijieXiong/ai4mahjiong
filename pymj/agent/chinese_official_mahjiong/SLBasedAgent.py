import torch
import os
import numpy as np
from collections import defaultdict

from pymj.agent.chinese_official_mahjiong.Agent import Agent
from pymj.dnn_model.resnet import PlayModel, FuroModel
from pymj.botzone.SLDataset1 import SLDataset
from pymj.agent.chinese_official_mahjiong.utils import choose_card2chi


class SLBasedAgent(Agent):
    DRAW_SOURCE = {
        "Play": 0,
        "Chi": 1,
        "Peng": 2,
        "Gang": 3,
        "BuGang": 4,
        "AnGang": 5,
        "Draw": 6
    }
    
    def __init__(self, random_generator: np.random.RandomState, play_model_path: str = None, chi_model_path: str = None, peng_model_path: str = None, gang_model_path: str = None, bu_gang_model_path: str = None, an_gang_model_path: str = None, device: str = "cpu", use_choose_card2_chi: bool = True):
        self.random_generator: np.random.RandomState = random_generator
        self.device: str = device
        self.play_model: PlayModel = None
        if play_model_path is not None:
            self.play_model = SLBasedAgent.load_model(play_model_path, "Play", self.device)
            self.play_model.eval()
        self.chi_model: FuroModel = None
        if chi_model_path is not None:
            self.chi_model = SLBasedAgent.load_model(chi_model_path, "Furo", self.device)
            self.chi_model.eval()
        self.peng_model: FuroModel = None
        if peng_model_path is not None:
            self.peng_model = SLBasedAgent.load_model(peng_model_path, "Furo", self.device)
            self.peng_model.eval()
        self.gang_model: FuroModel = None
        if gang_model_path is not None:
            self.gang_model = SLBasedAgent.load_model(gang_model_path, "Furo", self.device)
            self.gang_model.eval()
        self.bu_gang_model: FuroModel = None
        if bu_gang_model_path is not None:
            self.bu_gang_model = SLBasedAgent.load_model(bu_gang_model_path, "Furo", self.device)
            self.bu_gang_model.eval()
        self.an_gang_model: FuroModel = None
        if an_gang_model_path is not None:
            self.an_gang_model = SLBasedAgent.load_model(an_gang_model_path, "Furo", self.device)
            self.an_gang_model.eval()
        self.use_choose_card2_chi: bool = use_choose_card2_chi
            
    @staticmethod
    def load_model(model_path: str, model_type: str, device: str):
        assert model_type in ["Play", "Furo"]
        model_file_name: str = os.path.basename(model_path)
        num_layer: int = int(model_file_name.split("_")[1])
        if model_type == "Play":
            model: PlayModel = PlayModel(in_channels=28, num_layers=num_layer)
        else:
            model: FuroModel = FuroModel(in_channels=28, num_layers=num_layer)
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")["state_dict"]
        )
        return model.to(device)
        
    @staticmethod
    def feature_player_id(state: dict) -> list[list[int]]:
        features: list[int] = [0] * 35
        features[state["player_id"]] = 1
        return [features]
    
    @staticmethod
    def feature_wind(state: dict) -> list[list[int]]:
        game_wind: int = state["game_wind"]
        self_wind: int = state["self_wind"]
        features: list[int] = [0] * 35
        features[game_wind] = 1
        features[self_wind + 4] = 1
        return [features]
    
    @staticmethod
    def feature_self_cards(self_hand_card_ids: list[int]) -> list[list[int]]:
        exist_card_num: dict = defaultdict(int)
        features: list[list[int]] = [[0] * 35 for _ in range(4)]
        for card_id in self_hand_card_ids:
            features[exist_card_num[card_id]][card_id] = 1
            exist_card_num[card_id] += 1
        return features
    
    @staticmethod
    def feature_players_played_cards(state: dict) -> list[list[int]]:
        players_played_card_ids: tuple[list[int], ...] = state["players_played_card_ids"]
        features: list[list[int]] = []
        for player_played_card_ids in players_played_card_ids:
            player_features: list[int] = [0] * 35
            for card_id in player_played_card_ids:
                player_features[card_id] += 1
            features.append(player_features)
        return features
    
    @staticmethod
    def feature_players_melds(state: dict) -> list[list[int]]:
        players_melds: tuple[list[tuple[str, int, int]], ...] = state["players_melds"]
        features: list[list[int]] = []
        for player_melds in players_melds:
            player_features: list[int] = []
            for meld_type, _, meld_card_id in player_melds:
                player_feature = [0] * 35
                player_feature[34] = SLDataset.MELD_TYPE[meld_type]
                player_feature[meld_card_id] = 1
                player_features.append(player_feature)
            for _ in range(4 - len(player_melds)):
                player_features.append([0] * 35)
            features.extend(player_features)
        return features
    
    @staticmethod
    def festure_remain_cards(self_hand_card_ids: list[int], state: dict) -> list[list[int]]:
        players_played_card_ids: tuple[list[int], ...] = state["players_played_card_ids"]
        players_melds: tuple[list[tuple[str, int, int]], ...] = state["players_melds"]
        features: list[int] = [0] * 35
        exist_tiles = {i: 4 for i in range(34)}
        for player_melds in players_melds:
            for meld_type, _, meld_card_id in player_melds:
                if meld_type == "Chi":
                    exist_tiles[meld_card_id - 1] -= 1
                    exist_tiles[meld_card_id] -= 1
                    exist_tiles[meld_card_id + 1] -= 1
                elif meld_type == "Peng":
                    exist_tiles[meld_card_id] -= 3
                else:
                    exist_tiles[meld_card_id] -= 4
        for player_played_card_ids in players_played_card_ids:
            for player_played_card_id in player_played_card_ids:
                exist_tiles[player_played_card_id] -= 1
        for card_id in self_hand_card_ids:
            exist_tiles[card_id] -= 1
        for card_id, num_card in exist_tiles.items():
            for _ in range(num_card):
                features[card_id] += 1
        return [features]
    
    @staticmethod
    def get_common_features(self_hand_card_ids: list[int], state: dict) -> list[list[int]]:
        features: list[list[int]] = []
        features.extend(SLBasedAgent.feature_player_id(state))
        features.extend(SLBasedAgent.feature_self_cards(self_hand_card_ids))
        features.extend(SLBasedAgent.feature_players_melds(state))
        features.extend(SLBasedAgent.feature_players_played_cards(state))
        features.extend(SLBasedAgent.festure_remain_cards(self_hand_card_ids, state))
        features.extend(SLBasedAgent.feature_wind(state))
        return features
    
    @staticmethod
    def feature_last_action(state: dict) -> list[list[int]]:
        last_observation: tuple[int, str, int] = state["last_observation"]
        features = [0] * 35
        if len(last_observation[1]) > 0:
            features[SLBasedAgent.DRAW_SOURCE[last_observation[1]]] = 1

        return [features]

    def choose_play(self, self_hand_card_ids: list[int], state: dict) -> int:
        if self.play_model is None:
            return self.random_generator.choice(self_hand_card_ids)
        else:
            features: list[list[int]] = SLBasedAgent.get_common_features(self_hand_card_ids, state)
            features.extend(SLBasedAgent.feature_last_action(state))
            features = torch.tensor([features]).float().to(self.device)
            model_output: torch.Tensor = self.play_model(features)
            card2play_ids: list[int] = torch.sort(model_output[0, :-1], descending=True)[1].tolist()
            i: int = 0
            card2play_id: int = card2play_ids[i]
            while card2play_id not in self_hand_card_ids:
                i += 1
                card2play_id = card2play_ids[i]
            return card2play_id

    def choose_chi(self, self_hand_card_ids: list[int], state: dict, middle_card_ids: list[int]) -> int:
        if self.chi_model is None:
            return choose_card2chi(self_hand_card_ids, middle_card_ids) if \
                self.use_choose_card2_chi else self.random_generator.choice(middle_card_ids)
        else:
            features: list[list[int]] = SLBasedAgent.get_common_features(self_hand_card_ids, state)
            features.extend(SLBasedAgent.feature_last_action(state))
            features = torch.tensor([features]).float().to(self.device)
            do_chi = self.chi_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5
            if do_chi:
                return choose_card2chi(self_hand_card_ids, middle_card_ids) if \
                    self.use_choose_card2_chi else self.random_generator.choice(middle_card_ids)
            else:
                return -1

    def choose_peng(self, self_hand_card_ids: list[int], state: dict, card2peng_id: int) -> bool:
        if self.peng_model is None:
            return True
        else:
            features: list[list[int]] = SLBasedAgent.get_common_features(self_hand_card_ids, state)
            features.extend(SLBasedAgent.feature_last_action(state))
            features = torch.tensor([features]).float().to(self.device)
            return self.peng_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5

    def choose_gang(self, self_hand_card_ids: list[int], state: dict, card2gang_id: int) -> bool:
        if self.gang_model is None:
            return True
        else:
            features: list[list[int]] = SLBasedAgent.get_common_features(self_hand_card_ids, state)
            features.extend(SLBasedAgent.feature_last_action(state))
            features = torch.tensor([features]).float().to(self.device)
            return self.gang_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5

    def choose_bu_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        if self.bu_gang_model is None:
            return gang_card_ids[0]
        else:
            features: list[list[int]] = SLBasedAgent.get_common_features(self_hand_card_ids, state)
            card2gang_features: list[int] = [0] * 35
            for card_id in gang_card_ids:
                card2gang_features[card_id] = 1
            features.append(card2gang_features)
            features = torch.tensor(features).float().to(self.device)
            # todo: 改进，多个二分类组成多分类
            do_gang = self.bu_gang_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5
            if do_gang:
                return self.random_generator.choice(gang_card_ids)
            else:
                return -1

    def choose_an_gang(self, self_hand_card_ids: list[int], state: dict, gang_card_ids: list[int]) -> int:
        if self.an_gang_model is None:
            return gang_card_ids[0]
        else:
            features: list[list[int]] = SLBasedAgent.get_common_features(self_hand_card_ids, state)
            card2gang_features: list[int] = [0] * 35
            for card_id in gang_card_ids:
                card2gang_features[card_id] = 1
            features.append(card2gang_features)
            features = torch.tensor(features).float().to(self.device)
            do_gang = self.an_gang_model(features).squeeze(dim=-1).detach().cpu().item() > 0.5
            if do_gang:
                return self.random_generator.choice(gang_card_ids)
            else:
                return -1

    def choose_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True

    def choose_zi_mo_hu(self, self_hand_card_ids: list[int], state: dict, fan_result: list[tuple[int, str]]) -> bool:
        return True
