import json
import sys
from collections import Counter
from MahjongGB import MahjongFanCalculator

from action import *


def check_zi_mo_hu(state):
    self_hand_card_ids = state["self_hand_card_ids"]
    self_player_id = state["self_player_id"]
    self_wind = state["self_wind"]
    players_melds = state["players_melds"]
    last_observation = state["last_observation"]
    last_action_card_id = last_observation[2]
    game_wind = state["game_wind"]
    num_draw = state["num_draw"]
    players_played_card_ids = state["players_played_card_ids"]

    # MahjongFanCalculator需要的数据
    pack = []
    for meld_type, meld_player_id, meld_card_id in players_melds[self_player_id]:
        if meld_type == "Chi":
            claiming = (meld_type.upper(), card_int2str(meld_card_id), 1)
        else:
            claiming = (meld_type.replace("Bu", "").replace("An", "").upper(),
                        card_int2str(meld_card_id),
                        (self_player_id - meld_player_id + 4) % 4)
        pack.append(claiming)
    hand = tuple(map(card_int2str, self_hand_card_ids))
    winTile = card_int2str(last_action_card_id)
    flowerCount = 0
    isSelfDrawn = True
    played_card_ids = Counter(
        [card_id for player_played_card_ids in players_played_card_ids for card_id in player_played_card_ids])
    for i in range(4):
        for meld_type, _, meld_card_id in players_melds[i]:
            if meld_type == "Chi":
                played_card_ids[meld_card_id - 1] += 1
                played_card_ids[meld_card_id] += 1
                played_card_ids[meld_card_id + 1] += 1
            elif meld_type == "Peng":
                played_card_ids[meld_card_id] += 3
            else:
                played_card_ids[meld_card_id] += 4
    is4thTile = played_card_ids[last_action_card_id] == 3
    isAboutKong = False
    # 在botzone的规则下，每个人最多摸21次牌，自己的牌墙摸完即结束
    isWallLast = num_draw[self_wind] == 34
    seatWind = self_wind
    prevalentWind = game_wind
    verbose = False

    try:
        result = MahjongFanCalculator(pack, hand, winTile, flowerCount, isSelfDrawn, is4thTile,
                                      isAboutKong, isWallLast, seatWind, prevalentWind, verbose)
        fan_count = sum([res[0] for res in result])
        return result if fan_count >= 8 else []
    except:
        return []


def check_hu(state):
    self_hand_card_ids = state["self_hand_card_ids"]
    self_player_id = state["self_player_id"]
    self_wind = state["self_wind"]
    players_melds = state["players_melds"]
    last_action_player_id = state["last_observation"][0]
    last_action = state["last_observation"][1]
    last_action_card_id = state["last_observation"][2]
    game_wind = state["game_wind"]
    num_draw = state["num_draw"]
    players_played_card_ids = state["players_played_card_ids"]

    # MahjongFanCalculator需要的数据
    pack = []
    for meld_type, meld_player_id, meld_card_id in players_melds[self_player_id]:
        if meld_type == "Chi":
            claiming = (meld_type.upper(), card_int2str(meld_card_id), 1)
        else:
            claiming = (meld_type.replace("Bu", "").replace("An", "").upper(),
                        card_int2str(meld_card_id),
                        (self_player_id - meld_player_id + 4) % 4)
        pack.append(claiming)
    hand = tuple(map(card_int2str, self_hand_card_ids))
    winTile = card_int2str(last_action_card_id)
    flowerCount = 0
    isSelfDrawn = False

    played_card_ids = Counter(
        [card_id for player_played_card_ids in players_played_card_ids for card_id in player_played_card_ids])
    for i in range(4):
        for meld_type, _, meld_card_id in players_melds[i]:
            if meld_type == "Chi":
                played_card_ids[meld_card_id - 1] += 1
                played_card_ids[meld_card_id] += 1
                played_card_ids[meld_card_id + 1] += 1
            elif meld_type == "Peng":
                played_card_ids[meld_card_id] += 3
            else:
                played_card_ids[meld_card_id] += 4
    is4thTile = played_card_ids[last_action_card_id] == 3
    isAboutKong = last_action == "BuGang"
    # 在botzone的规则下，每个人最多摸21次牌，自己的牌墙摸完即结束
    isWallLast = num_draw[last_action_player_id] == 34
    seatWind = self_wind
    prevalentWind = game_wind
    verbose = False

    try:
        result = MahjongFanCalculator(tuple(pack), hand, winTile, flowerCount, isSelfDrawn, is4thTile,
                                      isAboutKong, isWallLast, seatWind, prevalentWind, verbose)
        fan_count = sum([res[0] for res in result])
        return result if fan_count >= 8 else []
    except:
        return []


def check_chi(state):
    # 0 2 2 2 3 7
    self_hand_card_ids = state["self_hand_card_ids"]
    # 6
    card2chi_id = state["last_observation"][2]
    chi_options = []

    if card2chi_id >= 27:
        return chi_options

    # 三种可能的顺子组合：
    # card2chi_id - 2, card2chi_id - 1, card2chi_id
    # card2chi_id - 1, card2chi_id, card2chi_id + 1
    # card2chi_id, card2chi_id + 1, card2chi_id + 2
    # 需要判断花色一致性（每种花色编号区间跨度为9）
    base = (card2chi_id // 9) * 9
    # 顺子: card2chi_id - 2, card2chi_id - 1, card2chi_id
    if card2chi_id - 2 >= base:
        if (card2chi_id - 2 in self_hand_card_ids) and (card2chi_id - 1 in self_hand_card_ids):
            chi_options.append(card2chi_id - 1)
    # 顺子: card2chi_id - 1, card2chi_id, card2chi_id + 1
    if (card2chi_id - 1 >= base) and (card2chi_id + 1 < base + 9):
        if (card2chi_id - 1 in self_hand_card_ids) and (card2chi_id + 1 in self_hand_card_ids):
            chi_options.append(card2chi_id)
    # 顺子: card2chi_id, card2chi_id + 1, card2chi_id + 2
    if card2chi_id + 2 < base + 9:
        if (card2chi_id + 1 in self_hand_card_ids) and (card2chi_id + 2 in self_hand_card_ids):
            chi_options.append(card2chi_id + 1)

    return chi_options


def check_peng(state):
    self_hand_card_ids = state["self_hand_card_ids"]
    card2peng_id = state["last_observation"][2]
    filtered_hand_cards = list(filter(lambda card_id: card_id == card2peng_id, self_hand_card_ids))
    return len(filtered_hand_cards) >= 2


def check_gang(state):
    self_hand_card_ids = state["self_hand_card_ids"]
    card2gang_id = state["last_observation"][2]
    filtered_hand_cards = list(filter(lambda card_id: card_id == card2gang_id, self_hand_card_ids))
    return len(filtered_hand_cards) >= 3


def check_bu_gang(state):
    self_hand_card_ids = state["self_hand_card_ids"]
    self_player_id = state["self_player_id"]
    players_melds = state["players_melds"]
    bu_gang_options = []

    for meld_type, _, meld_card_id in players_melds[self_player_id]:
        if meld_type == "Peng" and meld_card_id in self_hand_card_ids:
            bu_gang_options.append(meld_card_id)

    return bu_gang_options


def check_an_gang(state):
    counter = Counter(state["self_hand_card_ids"])
    return [elem for elem, count in counter.items() if count >= 4]


def card_str2int(tile_str):
    # T,B,W,F,J
    F, S = tile_str
    if F == "W":
        n = 0
    elif F == "T":
        n = 9
    elif F == "B":
        n = 18
    elif F == "F":
        n = 27
    else:
        n = 31
    return n + (int(S) - 1)


def card_int2str(card_id):
    assert 0 <= card_id <= 33
    F = card_id // 9
    S = card_id % 9
    if F == 0:
        return "W" + str(S + 1)
    elif F == 1:
        return "T" + str(S + 1)
    elif F == 2:
        return "B" + str(S + 1)
    else:
        if S < 4:
            return "F" + str(S + 1)
        else:
            return "J" + str(S - 3)


def response():
    data = {
        "players_melds": tuple([[] for _ in range(4)]),
        "players_played_card_actions": tuple([[] for _ in range(4)]),
        "players_played_card_ids": tuple([[] for _ in range(4)]),
        "num_draw": [13 for _ in range(4)]
    }
    full_input = json.loads(input())
    all_requests = full_input["requests"]
    current_request = all_requests[-1]
    request_contents = current_request.split(" ")
    request_id = request_contents[0]
    while int(request_id) in [0, 1, 2, 3]:
        state = {}
        if int(request_id) > 1:
            state["self_hand_card_ids"] = data["self_hand_card_ids"]
            state["self_player_id"] = data["self_wind"]
            state["self_wind"] = data["self_wind"]
            state["game_wind"] = data["game_wind"]
            state["players_melds"] = data["players_melds"]
            state["players_played_card_ids"] = data["players_played_card_ids"]

        if request_id == "0":
            # 门风和圈风信息
            data["self_wind"] = int(request_contents[1])
            data["game_wind"] = int(request_contents[2])
            my_action = "PASS"
        elif request_id == "1":
            # 手牌
            data["self_hand_card_ids"] = list(map(card_str2int, request_contents[5:18]))
            data["self_hand_card_ids"].sort()
            my_action = "PASS"
        elif request_id == "2":
            # 摸牌
            drawn_card_id = card_str2int(request_contents[1])
            data["self_hand_card_ids"].append(drawn_card_id)
            data["num_draw"][data["self_wind"]] += 1
            data["last_observation"] = (data["self_wind"], "Draw", drawn_card_id)
            state["last_observation"] = data["last_observation"]
            state["num_draw"] = data["num_draw"]
            my_action = response_self_draw(state)
            self_player_id = data["self_wind"]
            if "PLAY" in my_action:
                played_card_id = card_str2int(my_action.split(" ")[1])
                data["last_observation"] = (self_player_id, "Play", played_card_id)
                data["players_played_card_actions"][self_player_id].append("Play")
                data["players_played_card_ids"][self_player_id].append(played_card_id)
                data["self_hand_card_ids"].remove(played_card_id)
            elif "BUGANG" in my_action:
                bu_gang_card_id = card_str2int(my_action.split(" ")[1])
                data["last_observation"] = (self_player_id, "BuGang", bu_gang_card_id)
                self_melds = data["players_melds"][self_player_id]
                peng_idx = 0
                for i, (meld_action, _, meld_card_id) in enumerate(self_melds):
                    if meld_action == "Peng" and meld_card_id == bu_gang_card_id:
                        peng_idx = i
                        break
                self_melds[peng_idx] = ("BuGang", self_player_id, bu_gang_card_id)
                data["self_hand_card_ids"].remove(bu_gang_card_id)
            else:
                an_gang_card_id = card_str2int(my_action.split(" ")[1])
                data["last_observation"] = (self_player_id, "AnGang", an_gang_card_id)
                data["players_melds"][self_player_id].append(("AnGang", self_player_id, an_gang_card_id))
                data["self_hand_card_ids"].remove(an_gang_card_id)
                data["self_hand_card_ids"].remove(an_gang_card_id)
                data["self_hand_card_ids"].remove(an_gang_card_id)
        else:
            current_player_id = int(request_contents[1])
            current_player_action = request_contents[2]
            if current_player_id == data["self_wind"]:
                # 吃碰杠以及打牌都会发送给自己
                my_action = "PASS"
            elif current_player_action == "DRAW":
                # 其他人摸牌
                data["last_observation"] = (current_player_id, "Draw", -1)
                data["num_draw"][current_player_id] += 1
                my_action = "PASS"
            elif current_player_action in ["PLAY", "PENG", "CHI"]:
                last_player_id = data["last_observation"][0]
                if current_player_action == "PLAY":
                    # 其他人打出牌
                    played_card_id = card_str2int(request_contents[3])
                    data["last_observation"] = (current_player_id, "Play", played_card_id)
                    data["players_played_card_actions"][current_player_id].append("Play")
                    data["players_played_card_ids"][current_player_id].append(played_card_id)
                elif current_player_action == "PENG":
                    # 其他人碰牌并打出牌
                    _, _, peng_card_id = data["last_observation"]
                    played_card_id = card_str2int(request_contents[3])
                    data["players_played_card_actions"][last_player_id].pop()
                    data["players_played_card_ids"][last_player_id].pop()
                    data["last_observation"] = (current_player_id, "Play", played_card_id)
                    data["players_melds"][current_player_id].append(("Peng", last_player_id, peng_card_id))
                    data["players_played_card_actions"][current_player_id].append("Peng")
                    data["players_played_card_ids"][current_player_id].append(played_card_id)
                else:
                    # 其他人吃牌并打出牌
                    chi_middle_card_id = card_str2int(request_contents[3])
                    played_card_id = card_str2int(request_contents[4])
                    data["players_played_card_actions"][last_player_id].pop()
                    data["players_played_card_ids"][last_player_id].pop()
                    data["last_observation"] = (current_player_id, "Play", played_card_id)
                    data["players_melds"][current_player_id].append(("Chi", last_player_id, chi_middle_card_id))
                    data["players_played_card_actions"][current_player_id].append("Chi")
                    data["players_played_card_ids"][current_player_id].append(played_card_id)
                state["last_observation"] = data["last_observation"]
                state["players_melds"] = data["players_melds"]
                state["players_played_card_ids"] = data["players_played_card_ids"]
                state["num_draw"] = data["num_draw"]
                my_action = response_other_play(state)
                self_player_id = data["self_wind"]
                if "Gang" in my_action:
                    gang_card_id = card_str2int(my_action.split(" ")[1])
                    data["last_observation"] = (self_player_id, "Gang", gang_card_id)
                    data["players_melds"][self_player_id].append(("Gang", current_player_id, gang_card_id))
                    data["self_hand_card_ids"].remove(gang_card_id)
                    data["self_hand_card_ids"].remove(gang_card_id)
                    data["self_hand_card_ids"].remove(gang_card_id)
                    my_action = "GANG"
                elif "Peng" in my_action:
                    peng_card_id = card_str2int(my_action.split(" ")[1])
                    data["last_observation"] = (self_player_id, "Peng", peng_card_id)
                    data["players_melds"][self_player_id].append(("Peng", current_player_id, peng_card_id))
                    data["self_hand_card_ids"].remove(peng_card_id)
                    data["self_hand_card_ids"].remove(peng_card_id)
                    played_card_id = choose_card2play(state)
                    data["players_played_card_actions"][self_player_id].append("Peng")
                    data["players_played_card_ids"][self_player_id].append(played_card_id)
                    data["self_hand_card_ids"].remove(played_card_id)
                    my_action = f"PENG {card_int2str(played_card_id)}"
                elif "Chi" in my_action:
                    chi_card_id = data["last_observation"][2]
                    chi_middle_card_id = card_str2int(my_action.split(" ")[1])
                    data["last_observation"] = (self_player_id, "Chi", chi_middle_card_id)
                    data["players_melds"][self_player_id].append(("Chi", current_player_id, chi_middle_card_id))
                    if chi_card_id == (chi_middle_card_id - 1):
                        data["self_hand_card_ids"].remove(chi_middle_card_id)
                        data["self_hand_card_ids"].remove(chi_middle_card_id + 1)
                    elif chi_card_id == chi_middle_card_id:
                        data["self_hand_card_ids"].remove(chi_middle_card_id - 1)
                        data["self_hand_card_ids"].remove(chi_middle_card_id + 1)
                    else:
                        data["self_hand_card_ids"].remove(chi_middle_card_id - 1)
                        data["self_hand_card_ids"].remove(chi_middle_card_id)
                    played_card_id = choose_card2play(state)
                    data["players_played_card_actions"][self_player_id].append("Chi")
                    data["players_played_card_ids"][self_player_id].append(played_card_id)
                    data["self_hand_card_ids"].remove(played_card_id)
                    my_action = f"CHI {card_int2str(chi_middle_card_id)} {card_int2str(played_card_id)}"
            elif current_player_action == "BuGang":
                # 其他人补杠（可枪杆和）
                bu_gang_card_id = card_str2int(request_contents[3])
                other_melds = data["players_melds"][current_player_id]
                peng_idx = 0
                for i, (meld_action, _, meld_card_id) in enumerate(other_melds):
                    if meld_action == "Peng" and meld_card_id == bu_gang_card_id:
                        peng_idx = i
                        break
                other_melds[peng_idx] = ("BuGang", current_player_id, bu_gang_card_id)
                data["last_observation"] = (current_player_id, "BuGang", bu_gang_card_id)
                state["last_observation"] = data["last_observation"]
                state["num_draw"] = data["num_draw"]
                fan_result = check_hu(state)
                if len(fan_result) > 0 and choose_hu(state):
                    my_action = "HU"
                else:
                    my_action = "PASS"
            else:
                last_player_id, last_action, last_card_id = data["last_observation"]
                if last_action == "Play":
                    # 明杠
                    data["last_observation"] = (current_player_id, "Gang", last_card_id)
                    data["players_melds"][current_player_id].append(("Gang", last_player_id, last_card_id))
                    data["players_played_card_actions"][last_player_id].pop()
                    data["players_played_card_ids"][last_player_id].pop()
                else:
                    # 暗杠
                    data["last_observation"] = (current_player_id, "AnGang", -1)
                    data["players_melds"][current_player_id].append(("AnGang", current_player_id, -1))
                my_action = "PASS"
        print(json.dumps({
            "response": my_action,
        }))
        print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
        sys.stdout.flush()
        # input()会触发休眠，长时运行只支持简单模式
        current_request = input().strip()
        request_contents = current_request.split(" ")
        request_id = request_contents[0]


def response_self_draw(state):
    self_player_id = state["self_player_id"]
    num_draw = state["num_draw"]

    fan_result = check_zi_mo_hu(state)
    if len(fan_result) > 0 and choose_zi_mo_hu(state):
        return "HU"

    card_ids2an_gang = check_an_gang(state)
    if (len(card_ids2an_gang) > 0) and (num_draw[self_player_id] < 34) and choose_an_gang(state, card_ids2an_gang):
        card_id2an_gang = choose_card2an_gang(state, card_ids2an_gang)
        if card_id2an_gang >= 0:
            return f"GANG {card_int2str(card_id2an_gang)}"

    card_ids2bu_gang = check_bu_gang(state)
    if (len(card_ids2bu_gang) > 0) and (num_draw[self_player_id] < 34) and choose_bu_gang(state, card_ids2bu_gang):
        card_id2bu_gang = choose_card2bu_gang(state, card_ids2bu_gang)
        if card_id2bu_gang >= 0:
            return f"BUGANG {card_int2str(card_id2bu_gang)}"

    card_id2play = choose_card2play(state)
    return f"PLAY {card_int2str(card_id2play)}"


def response_other_play(state):
    self_player_id = state["self_player_id"]
    last_observation = state["last_observation"]
    num_draw = state["num_draw"]

    fan_result = check_hu(state)
    if len(fan_result) > 0 and choose_hu(state):
        return "HU"

    gang_card_id = last_observation[2]
    if check_gang(state) and (num_draw[self_player_id] < 34) and choose_gang(state, gang_card_id):
        return f"Gang {card_int2str(gang_card_id)}"

    peng_card_id = last_observation[2]
    if check_peng(state) and choose_peng(state, peng_card_id):
        return f"Peng {card_int2str(peng_card_id)}"

    last_player_id = last_observation[0]
    if self_player_id == (last_player_id + 1) % 4:
        middle_card_ids = check_chi(state)
        if len(middle_card_ids) > 0 and choose_chi(state, middle_card_ids):
            card_id2chi = choose_card2chi(state, middle_card_ids)
            return f"Chi {card_int2str(card_id2chi)}"

    return "PASS"
