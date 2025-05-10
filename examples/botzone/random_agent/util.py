import json
from MahjongGB import MahjongFanCalculator

from action import *


def check_zi_mo_hu(state: dict) -> list[tuple[int, str]]:
    self_hand_card_ids: list[int] = state["self_hand_card_ids"]
    self_player_id: int = state["self_player_id"]
    self_wind: int = state["self_wind"]
    players_melds: list[tuple] = state["players_melds"]
    last_observation: tuple = state["last_observation"]
    last_action_card_id: int = last_observation[2]
    is_last_card: bool = state["is_last_card"]
    game_wind: int = state["game_wind"]
    num_draw: list[int] = state["num_draw"]

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
    hand: tuple[str, ...] = tuple(map(card_int2str, self_hand_card_ids))
    winTile: str = card_int2str(last_action_card_id)
    flowerCount: int = 0
    isSelfDrawn: bool = True
    # 在botzone的规则下，每个人最多摸21次牌，自己的牌墙摸完即结束
    is4thTile: bool = num_draw[self_wind] == 21
    isAboutKong: bool = False
    isWallLast: bool = is_last_card
    seatWind: int = self_wind
    prevalentWind: int = game_wind
    verbose: bool = False

    try:
        result = MahjongFanCalculator(pack, hand, winTile, flowerCount, isSelfDrawn, is4thTile,
                                      isAboutKong, isWallLast, seatWind, prevalentWind, verbose)
        fan_count: int = sum([res[0] for res in result])
        return result if fan_count >= 8 else []
    except:
        return []


def check_hu(state: dict) -> list[tuple[int, str]]:
    self_hand_card_ids: list[int] = state["self_hand_card_ids"]
    self_player_id: int = state["self_player_id"]
    self_wind: int = state["self_wind"]
    players_melds: list[tuple] = state["players_melds"]
    last_action_player_id: int = state["last_observation"][0]
    last_action: str = state["last_observation"][1]
    last_action_card_id: int = state["last_observation"][2]
    is_last_card: bool = state["is_last_card"]
    game_wind: int = state["game_wind"]
    num_draw: list[int] = state["num_draw"]

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
    hand: tuple[str, ...] = tuple(map(card_int2str, self_hand_card_ids))
    winTile: str = card_int2str(last_action_card_id)
    flowerCount: int = 0
    isSelfDrawn: bool = False
    # 在botzone的规则下，每个人最多摸21次牌，自己的牌墙摸完即结束
    is4thTile: bool = num_draw[last_action_player_id] == 21
    isAboutKong: bool = last_action == "BuGang"
    isWallLast: bool = is_last_card
    seatWind: int = self_wind
    prevalentWind: int = game_wind
    verbose: bool = False

    try:
        result = MahjongFanCalculator(tuple(pack), hand, winTile, flowerCount, isSelfDrawn, is4thTile,
                                      isAboutKong, isWallLast, seatWind, prevalentWind, verbose)
        fan_count: int = sum([res[0] for res in result])
        return result if fan_count >= 8 else []
    except:
        return []


def check_chi(state: dict) -> list[int]:
    self_hand_card_ids: list[int] = state["self_hand_card_ids"]
    card2chi_id: int = state["last_observation"][2]
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


def check_peng(state: dict) -> int:
    self_hand_card_ids: list[int] = state["self_hand_card_ids"]
    card2peng_id: int = state["last_observation"][2]
    filtered_hand_cards = list(filter(lambda card_id: card_id == card2peng_id, self_hand_card_ids))
    return card2peng_id if len(filtered_hand_cards) >= 2 else -1


def check_gang(state: dict) -> int:
    self_hand_card_ids: list[int] = state["self_hand_card_ids"]
    card2gang_id: int = state["last_observation"][2]
    filtered_hand_cards: list = list(filter(lambda card_id: card_id == card2gang_id, self_hand_card_ids))
    return card2gang_id if len(filtered_hand_cards) >= 3 else -1


def check_bu_gang(state: dict) -> list:
    self_hand_card_ids: list[int] = state["self_hand_card_ids"]
    self_player_id: int = state["self_player_id"]
    players_melds: list[tuple] = state["players_melds"]
    bu_gang_options = []

    for i, (meld_type, _, meld_card_id) in enumerate(players_melds[self_player_id]):
        if meld_type == "Peng" and meld_card_id in self_hand_card_ids:
            bu_gang_options.append(i)

    return bu_gang_options


def check_an_gang(state: dict) -> list:
    counter = Counter(state["self_hand_card_ids"])
    return [elem for elem, count in counter.items() if count >= 4]


def card_str2int(tile_str: str) -> int:
    # T,B,W,F,J
    F, S = tile_str
    if F == "T":
        n = 0
    elif F == "B":
        n = 9
    elif F == "W":
        n = 18
    elif F == "F":
        n = 27
    else:
        n = 31
    return n + (int(S) - 1)


def card_int2str(card_id: int) -> str:
    assert 0 <= card_id <= 33
    F: int = card_id // 9
    S: int = card_id % 9
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


# self_hand_card_ids: list[int] = state["self_hand_card_ids"]
# self_player_id: int = state["self_player_id"]
# self_wind: int = state["self_wind"]
# players_melds: list[tuple] = state["players_melds"]
# last_observation: tuple = state["last_observation"]
# last_action_card_id: int = last_observation[2]
# players_played_card_ids: tuple[list] = state["players_played_card_ids"]
# is_last_card: bool = state["is_last_card"]
# game_wind: int = state["game_wind"]
def parse_json_input():
    full_input = json.loads(input())
    if "data" not in full_input:
        full_input["data"] = {
            "players_melds": tuple([[] for _ in range(4)]),
            "players_played_card_actions": tuple([[] for _ in range(4)]),
            "players_played_card_ids": tuple([[] for _ in range(4)]),
            "num_draw": [0 for _ in range(4)]
        }
    data = full_input["data"]

    all_requests = full_input["requests"]
    current_request = all_requests[-1]
    request_contents = current_request.split(" ")
    request_id = request_contents[0]
    state = {}
    if int(request_id) > 1:
        state = {
            "self_hand_card_ids": data["self_hand_card_ids"],
            "self_player_id": data["self_wind"],
            "self_wind": data["self_wind"],
            "game_wind": data["game_wind"],
        }

    if request_id == "0":
        # 门风和圈风信息
        data["self_wind"] = int(request_contents[1])
        data["game_wind"] = int(request_contents[2])
        my_action = "PASS"
    elif request_id == "1":
        # 手牌
        data["self_hand_card_ids"] = list(map(card_str2int, request_contents[5:18]))
        my_action = "PASS"
    elif request_id == "2":
        # 摸牌
        drawn_card_id = card_str2int(request_contents[1])
        data["self_hand_card_ids"].append(drawn_card_id)
        data["num_draw"][data["self_wind"]] += 1
        state["last_observation"] = (data["self_player_id"], "Draw", drawn_card_id)
        state["num_draw"] = data["num_draw"]
        my_action = response_self_draw(state)
    else:
        current_player_id = int(request_contents[1])
        current_player_action = request_contents[2]
        if current_player_action == "DRAW":
            # 其他人摸牌
            state["last_observation"] = (current_player_id, "Draw", -1)
            data["num_draw"][current_player_id] += 1
            state["num_draw"] = data["num_draw"]
            my_action = "PASS"
        elif current_player_action in ["PLAY", "PENG", "CHI"]:
            if current_player_action == "PLAY":
                # 其他人打出牌
                played_card_id = card_str2int(request_contents[3])
                state["last_observation"] = (current_player_id, "Play", played_card_id)
                data["players_played_card_actions"][current_player_id].append("Play")
            elif current_player_action == "PENG":
                # 其他人碰牌并打出牌
                peng_card_id = card_str2int(request_contents[3])
                played_card_id = card_str2int(request_contents[4])
                state["last_observation"] = (current_player_id, "Play", played_card_id)
                data["players_melds"][current_player_id].append("Peng", current_player_id, peng_card_id)
                data["players_played_card_actions"][current_player_id].append("Peng")
            else:
                # 其他人吃牌并打出牌
                chi_middle_card_id = card_str2int(request_contents[3])
                played_card_id = card_str2int(request_contents[4])
                state["last_observation"] = (current_player_id, "Play", played_card_id)
                data["players_melds"][current_player_id].append("Chi", current_player_id, chi_middle_card_id)
                data["players_played_card_actions"][current_player_id].append("Chi")
            data["players_played_card_ids"][current_player_id].append(played_card_id)
            state["players_melds"] = data["players_melds"]
            state["players_played_card_ids"] = data["players_played_card_ids"]
            state["num_draw"] = data["num_draw"]
            my_action = response_other_play(state)
        elif current_player_action == "BuGang":
            # 其他人补杠（可枪杆和）
            bu_gang_card_id = card_str2int(request_contents[3])
            state["last_observation"] = (current_player_id, "BuGang", bu_gang_card_id)
            state["num_draw"] = data["num_draw"]
            fan_result = check_hu(state)
            if len(fan_result) > 0 and choose_hu(state):
                my_action = "Hu"
            else:
                my_action = "PASS"
        else:
            # GANG,明杠或者暗杆，需要自己判断
            # todo: 更新各玩家副露信息
            my_action = "PASS"
    print(json.dumps({
        "response": my_action,
        # 可以存储一些前述的信息，在该对局下回合中使用，可以是dict或者字符串
        "data": data
    }))


def response_self_draw(state):
    fan_result: list[tuple[int, str]] = check_zi_mo_hu(state)
    if len(fan_result) > 0 and choose_zi_mo_hu(state):
        # 判读是否胡牌
        return "Hu"
    card_ids2an_gang: list[int] = check_an_gang(state)
    if len(card_ids2an_gang) > 0 and choose_an_gang(state, card_ids2an_gang):
        # 判断是否杠
        card_id2an_gang = choose_card2an_gang(state, card_ids2an_gang)
        if card_id2an_gang >= 0:
            return f"GANG {card_id2an_gang}"
    card_ids2bu_gang: list[int] = check_bu_gang(state)
    if len(card_ids2bu_gang) > 0 and choose_bu_gang(state, card_ids2bu_gang):
        # 判断是否杠
        card_id2bu_gang = choose_card2bu_gang(state, card_ids2an_gang)
        if card_id2bu_gang >= 0:
            return f"GANG {card_id2bu_gang}"
    # 选择一张牌打出去
    card_id2play: int = choose_card2play(state)
    return f"PLAY {card_int2str(card_id2play)}"


def response_other_play(state):
    self_player_id: int = state["self_player_id"]
    last_observation: tuple = state["last_observation"]
    fan_result = check_hu(state)
    if len(fan_result) > 0 and choose_hu(state):
        # 判读是否胡牌
        return "Hu"
    card2gang_id = check_gang(state)
    if card2gang_id != -1:
        # 判断是否杠
        pass
    card2peng_id = check_peng(state)
    if card2peng_id != -1:
        # 判断是否碰
        pass
    if self_player_id == (last_observation[0] + 1) % 4:
        middle_card_ids = check_chi(state)
        if len(middle_card_ids) > 0 and choose_chi(state, middle_card_ids):
            # 判断是否吃
            pass


def response_other_bu_gang(state):
    pass
