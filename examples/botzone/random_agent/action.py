import random
from collections import Counter


def choose_hu(state: dict) -> bool:
    return True


def choose_zi_mo_hu(state: dict) -> bool:
    return True


def choose_chi(state: dict, middle_card_ids: list[int]) -> bool:
    return True


def choose_peng(state: dict, card2peng_id: int) -> bool:
    return True


def choose_gang(state: dict, card2gang_id: int) -> bool:
    return True


def choose_an_gang(state: dict, gang_card_ids: list[int]) -> bool:
    return True


def choose_bu_gang(state: dict, gang_card_ids: list[int]) -> bool:
    return True


def get_suit_and_num(card_id):
    if 0 <= card_id <= 8:
        return '万', card_id + 1
    elif 9 <= card_id <= 17:
        return '条', card_id - 8
    elif 18 <= card_id <= 26:
        return '饼', card_id - 17
    else:
        return '字', card_id


def choose_card2play(state: dict) -> int:
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


def count_partner(hand_counter: Counter) -> int:
    """估算搭子数（顺子连张和对子）"""
    partner = 0
    for base in [0, 9, 18]:  # 万/条/饼
        tiles = [hand_counter[base + i] for i in range(9)]
        i = 0
        while i < 9:
            # 对子
            if tiles[i] >= 2:
                partner += 1
                tiles[i] -= 2
                continue
            # 连张
            if i <= 7 and tiles[i] >= 1 and tiles[i+1] >= 1:
                partner += 1
                tiles[i] -= 1
                tiles[i+1] -= 1
                continue
            i += 1
    return partner


def choose_card2chi(state: dict, middle_card_ids: list[int]) -> int:
    self_hand_card_ids = state["self_hand_card_ids"]
    best_choice = middle_card_ids[0]
    max_partner = -1

    original_counter = Counter(self_hand_card_ids)

    for chi_mid in middle_card_ids:
        # 计算要吃的三张
        idx = chi_mid % 9

        # 确定吃的范围：中张在0~8中，只能吃idx-1, idx, idx+1
        chi_group = [chi_mid - 1, chi_mid, chi_mid + 1]

        # 模拟吃牌（移除手牌中拿到的两张）
        new_counter = original_counter.copy()
        new_counter[chi_group[0]] -= 1
        new_counter[chi_group[1]] -= 1
        # chi_group[2]是上家打出的，不从手牌里扣

        # 注意：不实际删0或负数，只用于搭子估算
        partner = count_partner(new_counter)

        if partner > max_partner:
            max_partner = partner
            best_choice = chi_mid

    return best_choice


def choose_card2an_gang(state: dict, gang_card_ids: list[int]) -> int:
    return random.choice(gang_card_ids)


def choose_card2bu_gang(state: dict, gang_card_ids: list[int]) -> int:
    return random.choice(gang_card_ids)
