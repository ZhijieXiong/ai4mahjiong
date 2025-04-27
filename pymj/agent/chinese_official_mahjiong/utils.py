from collections import Counter


def choose_card2chi(self_hand_card_ids: list[int], middle_card_ids: list[int]) -> int:
    def count_taatsu(hand_counter: Counter) -> int:
        """估算搭子数（顺子连张和对子）"""
        taatsu = 0
        for base in [0, 9, 18]:  # 万/条/饼
            tiles = [hand_counter[base + i] for i in range(9)]
            i = 0
            while i < 9:
                # 对子
                if tiles[i] >= 2:
                    taatsu += 1
                    tiles[i] -= 2
                    continue
                # 连张
                if i <= 7 and tiles[i] >= 1 and tiles[i+1] >= 1:
                    taatsu += 1
                    tiles[i] -= 1
                    tiles[i+1] -= 1
                    continue
                i += 1
        return taatsu

    best_choice = middle_card_ids[0]
    max_taatsu = -1

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

        taatsu = count_taatsu(new_counter)

        if taatsu > max_taatsu:
            max_taatsu = taatsu
            best_choice = chi_mid

    return best_choice
