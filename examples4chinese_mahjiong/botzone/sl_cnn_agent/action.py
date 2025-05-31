import random
import torch
from collections import defaultdict, Counter

from model import load_model

MODELS_PATH = {
    "Chi": "data/SL_models/Chi_10_0.0001_1024_1e-06.ckt",
    "Peng": "data/SL_models/Peng_7_0.0001_1024_1e-06.ckt",
    "AnGang": "data/SL_models/AnGang_5_0.0001_1024_1e-06.ckt",
    "BuGang": "data/SL_models/BuGang_5_0.0001_1024_1e-06.ckt",
    "Gang": "data/SL_models/Gang_5_0.0001_1024_1e-06.ckt",
    "Play": "data/SL_models/Play_10_0.0001_256_1e-06.ckt"
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLAY_MODEL = load_model(MODELS_PATH["Play"], "Play", DEVICE)
CHI_MODEL = load_model(MODELS_PATH["Chi"], "Furo", DEVICE)
PENG_MODEL = load_model(MODELS_PATH["Peng"], "Furo", DEVICE)
GANG_MODEL = load_model(MODELS_PATH["Gang"], "Furo", DEVICE)
AN_GANG_MODEL = load_model(MODELS_PATH["AnGang"], "Furo", DEVICE)
BU_GANG_MODEL = load_model(MODELS_PATH["BuGang"], "Furo", DEVICE)
PLAY_MODEL.eval()
CHI_MODEL.eval()
PENG_MODEL.eval()
GANG_MODEL.eval()
AN_GANG_MODEL.eval()
BU_GANG_MODEL.eval()
PENG_MODEL.eval()
DRAW_SOURCE = {
    "Play": 0,
    "Chi": 1,
    "Peng": 2,
    "Gang": 3,
    "BuGang": 4,
    "AnGang": 5,
    "Draw": 6
}
MELD_TYPE = {
    "Chi": 0,
    "Peng": 1,
    "Gang": 2,
    "BuGang": 3,
    "AnGang": 4
}


# ==========================================特征编码模块==========================================
def feature_player_id(state):
    features = [0] * 35
    features[state["self_player_id"]] = 1
    return [features]


def feature_wind(state):
    game_wind = state["game_wind"]
    self_wind = state["self_wind"]
    features = [0] * 35
    features[game_wind] = 1
    features[self_wind + 4] = 1
    return [features]


def feature_self_cards(self_hand_card_ids):
    exist_card_num = defaultdict(int)
    features = [[0] * 35 for _ in range(4)]
    for card_id in self_hand_card_ids:
        features[exist_card_num[card_id]][card_id] = 1
        exist_card_num[card_id] += 1
    return features


def feature_players_played_cards(state):
    players_played_card_ids = state["players_played_card_ids"]
    features = []
    for player_played_card_ids in players_played_card_ids:
        player_features = [0] * 35
        for card_id in player_played_card_ids:
            player_features[card_id] += 1
        features.append(player_features)
    return features


def feature_players_melds(state):
    players_melds = state["players_melds"]
    features = []
    for player_melds in players_melds:
        player_features = []
        for meld_type, _, meld_card_id in player_melds:
            player_feature = [0] * 35
            player_feature[34] = MELD_TYPE[meld_type]
            if meld_type != "AnGang":
                player_feature[meld_card_id] = 1
            else:
                player_feature[34] = 1
            player_features.append(player_feature)
        for _ in range(4 - len(player_melds)):
            player_features.append([0] * 35)
        features.extend(player_features)
    return features


def feature_remain_cards(self_hand_card_ids, state):
    players_played_card_ids = state["players_played_card_ids"]
    players_melds = state["players_melds"]
    features = [0] * 35
    exist_tiles = {i: 4 for i in range(34)}
    for player_melds in players_melds:
        for meld_type, _, meld_card_id in player_melds:
            if meld_type == "Chi":
                exist_tiles[meld_card_id - 1] -= 1
                exist_tiles[meld_card_id] -= 1
                exist_tiles[meld_card_id + 1] -= 1
            elif meld_type == "Peng":
                exist_tiles[meld_card_id] -= 3
            elif meld_type in ["Gang", "BuGang"]:
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


def get_common_features(self_hand_card_ids, state):
    features = []
    features.extend(feature_player_id(state))
    features.extend(feature_self_cards(self_hand_card_ids))
    features.extend(feature_players_melds(state))
    features.extend(feature_players_played_cards(state))
    features.extend(feature_remain_cards(self_hand_card_ids, state))
    features.extend(feature_wind(state))
    return features


def feature_last_action(state):
    last_observation = state["last_observation"]
    features = [0] * 35
    if len(last_observation[1]) > 0:
        features[DRAW_SOURCE[last_observation[1]]] = 1

    return [features]


# ================================================================================================


# ==========================================决定是否做动作==========================================
def choose_hu(state):
    return True


def choose_zi_mo_hu(state):
    return True


def choose_chi(state, middle_card_ids):
    try:
        # todo: 有bug，RuntimeError: Given groups=1, weight of size [256, 28, 3], expected input[1, 29, 35] to have 28 channels, but got 29 channels instead
        features = get_common_features(state["self_hand_card_ids"], state)
        features.extend(feature_last_action(state))
        features = torch.tensor([features]).float().to(DEVICE)
        return CHI_MODEL(features).squeeze(dim=-1).detach().cpu().item() > 0.5
    except:
        return True


def choose_peng(state, card2peng_id):
    try:
        features = get_common_features(state["self_hand_card_ids"], state)
        features.extend(feature_last_action(state))
        features = torch.tensor([features]).float().to(DEVICE)
        return PENG_MODEL(features).squeeze(dim=-1).detach().cpu().item() > 0.5
    except:
        return True


def choose_gang(state, card2gang_id):
    try:
        features = get_common_features(state["self_hand_card_ids"], state)
        features.extend(feature_last_action(state))
        features = torch.tensor([features]).float().to(DEVICE)
        return GANG_MODEL(features).squeeze(dim=-1).detach().cpu().item() > 0.5
    except:
        return True


def choose_an_gang(state, gang_card_ids):
    # todo: 有bug, RuntimeError: Expected 3-dimensional input for 3-dimensional weight[256, 28, 3], but got 2-dimensional input of size [28, 35] instead
    try:
        features = get_common_features(state["self_hand_card_ids"], state)
        card2gang_features = [0] * 35
        for card_id in gang_card_ids:
            card2gang_features[card_id] = 1
        features.append(card2gang_features)
        features = torch.tensor(features).float().to(DEVICE)
        return AN_GANG_MODEL(features).squeeze(dim=-1).detach().cpu().item() > 0.5
    except:
        return True


def choose_bu_gang(state, gang_card_ids):
    # todo: 估计也有bug，但是还没测试到
    try:
        features = get_common_features(state["self_hand_card_ids"], state)
        card2gang_features: list[int] = [0] * 35
        for card_id in gang_card_ids:
            card2gang_features[card_id] = 1
        features.append(card2gang_features)
        features = torch.tensor(features).float().to(DEVICE)
        return BU_GANG_MODEL(features).squeeze(dim=-1).detach().cpu().item() > 0.5
    except:
        return True


# ================================================================================================


# ==========================================决定动作的具体值=========================================
def get_suit_and_num(card_id):
    if 0 <= card_id <= 8:
        return '万', card_id + 1
    elif 9 <= card_id <= 17:
        return '条', card_id - 8
    elif 18 <= card_id <= 26:
        return '饼', card_id - 17
    else:
        return '字', card_id


def choose_card2play(state):
    try:
        self_hand_card_ids = state["self_hand_card_ids"]
        features = get_common_features(self_hand_card_ids, state)
        features.extend(feature_last_action(state))
        features = torch.tensor([features]).float().to(DEVICE)
        model_output: torch.Tensor = PLAY_MODEL(features)
        card2play_ids = torch.sort(model_output[0, :-1], descending=True)[1].tolist()
        i = 0
        card2play_id = card2play_ids[i]
        while card2play_id not in self_hand_card_ids:
            i += 1
            card2play_id = card2play_ids[i]
        return card2play_id
    except:
        self_hand_card_ids = state["self_hand_card_ids"]
        counts = Counter(self_hand_card_ids)
        suit_groups = {'万': [], '条': [], '饼': []}
        for c in self_hand_card_ids:
            s, n = get_suit_and_num(c)
            if s in suit_groups:
                suit_groups[s].append(n)

        min_value = float('inf')
        candidates = []
        for card_id in self_hand_card_ids:
            s, num = get_suit_and_num(card_id)
            if s in suit_groups:  # 数牌
                suit_nums = suit_groups[s]
                left = (num - 1) in suit_nums
                right = (num + 1) in suit_nums
                # 基础分
                if num in (1, 9):
                    value = 0
                elif num in (2, 8):
                    value = 1
                else:
                    value = 2
                # 对子/刻子加分
                current_count = counts[card_id]
                if current_count >= 2:
                    value += 2 * (current_count - 1)
                # 相邻加分
                if left:
                    value += 1
                if right:
                    value += 1
            else:  # 字牌
                current_count = counts[card_id]
                value = current_count

            if value < min_value:
                min_value = value
                candidates = [card_id]
            elif value == min_value:
                candidates.append(card_id)

        return random.choice(candidates)


def choose_card2chi(state, middle_card_ids):
    return random.choice(middle_card_ids)


def choose_card2an_gang(state, gang_card_ids):
    return random.choice(gang_card_ids)


def choose_card2bu_gang(state, gang_card_ids):
    return random.choice(gang_card_ids)
# ================================================================================================
