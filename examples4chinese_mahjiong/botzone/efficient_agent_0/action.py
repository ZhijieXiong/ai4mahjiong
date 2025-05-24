import random
from collections import Counter


# ==========================================决定是否做动作==========================================
def choose_hu(state):
    return True


def choose_zi_mo_hu(state):
    return True


def choose_chi(state, middle_card_ids):
    return True


def choose_peng(state, card2peng_id):
    return True


def choose_gang(state, card2gang_id):
    return True


def choose_an_gang(state, gang_card_ids):
    return True


def choose_bu_gang(state, gang_card_ids):
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
