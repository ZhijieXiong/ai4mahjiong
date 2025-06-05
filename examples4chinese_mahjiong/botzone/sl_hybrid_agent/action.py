import random
import torch
from collections import defaultdict, Counter

from model import load_model

MODELS_PATH = {
    "Chi": "data/SL_hybrid_models/Chi.ckt",
    "Peng": "data/SL_hybrid_models/Peng.ckt",
    "Gang": "data/SL_hybrid_models/Gang.ckt",
    "Play": "data/SL_hybrid_models/Play.ckt"
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLAY_MODEL = load_model(MODELS_PATH["Play"], "Play", DEVICE)
CHI_MODEL = load_model(MODELS_PATH["Chi"], "Chi", DEVICE)
PENG_MODEL = load_model(MODELS_PATH["Peng"], "Peng", DEVICE)
GANG_MODEL = load_model(MODELS_PATH["Gang"], "Gang", DEVICE)
PLAY_MODEL.eval()
CHI_MODEL.eval()
PENG_MODEL.eval()
GANG_MODEL.eval()


# ==========================================特征编码模块==========================================
def feature_self_cards(self_hand_card_ids):
    exist_card_num = defaultdict(int)
    features4mlp = [0] * 34
    features4cnn = [[0] * 34 for _ in range(4)]
    for card_id in self_hand_card_ids:
        features4mlp[card_id] += 1
        features4cnn[exist_card_num[card_id]][card_id] = 1
        exist_card_num[card_id] += 1
    return torch.tensor(features4mlp).float(), torch.tensor(features4cnn).float()


def feature_melds(melds):
    melds_features = []
    for meld_type, _, meld_card_id in melds:
        meld_features = [0] * 5 + [-1] * 4
        if meld_type == "Chi":
            meld_features[0] = 1
            meld_features[5] = meld_card_id - 1
            meld_features[6] = meld_card_id
            meld_features[7] = meld_card_id + 1
        elif meld_type == "Peng":
            meld_features[1] = 1
            meld_features[5:8] = [meld_card_id] * 3
        else:
            if meld_type == "Gang":
                meld_features[2] = 1
            elif meld_type == "AnGang":
                meld_features[3] = 1
            else:
                meld_features[4] = 1
            meld_features[5:9] = [meld_card_id] * 4
        melds_features.append(meld_features)
    while len(melds_features) < 4:
        melds_features.append([0] * 5 + [-1] * 4)
    random.shuffle(melds_features)
    return torch.flatten(torch.tensor(melds_features)).float()


def feature_played_cards(
        played_cards, num_history
):
    padding_len = max(0, num_history - len(played_cards))
    features4rnn = [card_id + 1 for card_id in played_cards[-num_history:]] + [0] * padding_len
    features4mlp = [0] * 34
    for card_id in played_cards:
        features4mlp[card_id] += 1
    return (torch.tensor(features4mlp).float(),
            torch.tensor(features4rnn).long(),
            torch.tensor(min(num_history, len(played_cards))).long())


def feature_wind(self_wind, game_wind):
    features = [0] * 8
    features[self_wind] = 1
    features[4 + game_wind] = 1
    return torch.tensor(features).long()


def feature_remain_cards(
        self_player_id,
        self_hand_card_ids,
        players_played_card_ids,
        players_melds
):
    features = [0] * 34
    exist_tiles = {i: 4 for i in range(34)}
    for player_id, player_melds in enumerate(players_melds):
        for meld_type, _, meld_card_id in player_melds:
            if meld_type == "Chi":
                exist_tiles[meld_card_id - 1] -= 1
                exist_tiles[meld_card_id] -= 1
                exist_tiles[meld_card_id + 1] -= 1
            elif meld_type == "Peng":
                exist_tiles[meld_card_id] -= 3
            elif meld_type in ["Gang", "BuGang"]:
                exist_tiles[meld_card_id] -= 4
            else:
                if player_id == self_player_id:
                    exist_tiles[meld_card_id] -= 4
    for player_played_card_ids in players_played_card_ids:
        for player_played_card_id in player_played_card_ids:
            exist_tiles[player_played_card_id] -= 1
    for card_id in self_hand_card_ids:
        exist_tiles[card_id] -= 1
    for card_id, num_card in exist_tiles.items():
        for _ in range(num_card):
            features[card_id] += 1
    return torch.tensor(features).float()


def get_features(self_hand_card_ids, state):
    self_wind = int(state["self_wind"])
    self_player_id = self_wind
    game_wind = int(state["game_wind"])
    players_melds = state["players_melds"]
    players_played_card_ids = state["players_played_card_ids"]
    wind_features = feature_wind(self_wind, game_wind)
    self_hand_card_features4mlp, features4cnn = feature_self_cards(
        self_hand_card_ids)
    self_played_card_features4mlp, self_played_card_features4rnn, self_seq_len = (
        feature_played_cards(players_played_card_ids[self_player_id], 21))
    self_melds_features = feature_melds(players_melds[self_player_id])
    other_played_card_ids = []
    for i in range(4):
        if i != self_player_id:
            other_played_card_ids.extend(players_played_card_ids[i])
    other_played_card_features4mlp, _, _ = feature_played_cards(other_played_card_ids, 21)
    left_player_id = (self_player_id - 1) if (self_player_id > 0) else 3
    right_player_id = (self_player_id + 1) if (self_player_id < 3) else 0
    across_player_id = (self_player_id + 2) if (self_player_id < 2) else (self_player_id - 2)
    _, left_played_card_features4rnn, left_seq_len = (feature_played_cards(players_played_card_ids[left_player_id], 21))
    _, right_played_card_features4rnn, right_seq_len = (
        feature_played_cards(players_played_card_ids[right_player_id], 21))
    _, across_played_card_features4rnn, across_seq_len = (
        feature_played_cards(players_played_card_ids[across_player_id], 21))
    remain_card_features = feature_remain_cards(self_player_id, self_hand_card_ids, tuple(players_played_card_ids),
                                                players_melds)
    left_melds_features = feature_melds(players_melds[left_player_id])
    right_melds_features = feature_melds(players_melds[right_player_id])
    across_melds_features = feature_melds(players_melds[across_player_id])
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
    return features4mlp.to(DEVICE), features4cnn.to(DEVICE), features4rnn.to(DEVICE), rnn_seqs_len.to(DEVICE)


# ================================================================================================


# ==========================================决定是否做动作==========================================
def choose_hu(state):
    return True


def choose_zi_mo_hu(state):
    return True


def choose_chi(state, middle_card_ids):
    try:
        self_hand_card_ids = state["self_hand_card_ids"]
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = get_features(self_hand_card_ids, state)
        return CHI_MODEL(features4mlp, features4cnn, features4rnn, rnn_seqs_len).squeeze(
            dim=-1).detach().cpu().item() > 0.5
    except:
        return False


def choose_peng(state, card2peng_id):
    try:
        self_hand_card_ids = state["self_hand_card_ids"]
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = get_features(self_hand_card_ids, state)
        return PENG_MODEL(features4mlp, features4cnn, features4rnn, rnn_seqs_len).squeeze(
            dim=-1).detach().cpu().item() > 0.5
    except:
        return False


def choose_gang(state, card2gang_id):
    try:
        self_hand_card_ids = state["self_hand_card_ids"]
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = get_features(self_hand_card_ids, state)
        model_output = GANG_MODEL(features4mlp, features4cnn, features4rnn, rnn_seqs_len)
        return (model_output[0][2] / model_output[0][3]) >= 1
    except:
        return False


def choose_an_gang(state, gang_card_ids):
    try:
        self_hand_card_ids = state["self_hand_card_ids"]
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = get_features(self_hand_card_ids, state)
        model_output = GANG_MODEL(features4mlp, features4cnn, features4rnn, rnn_seqs_len)
        return (model_output[0][1] / model_output[0][3]) >= 1
    except:
        return False


def choose_bu_gang(state, gang_card_ids):
    try:
        self_hand_card_ids = state["self_hand_card_ids"]
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = get_features(self_hand_card_ids, state)
        model_output = GANG_MODEL(features4mlp, features4cnn, features4rnn, rnn_seqs_len)
        return (model_output[0][0] / model_output[0][3]) >= 1
    except:
        return False


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
        features4mlp, features4cnn, features4rnn, rnn_seqs_len = get_features(self_hand_card_ids, state)
        model_output = PLAY_MODEL(features4mlp, features4cnn, features4rnn, rnn_seqs_len)
        card2play_ids = torch.sort(model_output[0], descending=True)[1].tolist()
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
