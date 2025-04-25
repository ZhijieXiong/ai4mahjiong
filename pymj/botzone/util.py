import os
import random
from enum import Enum, auto


def split_data(botzone_data_path, target_dir):
    all_dir = os.path.join(target_dir, "all")
    train_dir = os.path.join(target_dir, "train")
    valid_dir = os.path.join(target_dir, "valid")
    test_dir = os.path.join(target_dir, "test")
    if not os.path.exists(all_dir):
        os.mkdir(all_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    one_hand_data = ""
    hand_id = ""
    data_all = []
    with open(botzone_data_path, "r") as rf:
        for line in rf.readlines():
            if "Match" in line:
                if len(hand_id) > 0:
                    target_path = os.path.join(all_dir, hand_id) + ".txt"
                    with open(target_path, "w") as wf:
                        wf.write(one_hand_data)
                    print(f"{hand_id} done")
                    data_all.append((hand_id, one_hand_data))
                one_hand_data = ""
                hand_id = line.split(" ")[1].strip()
            else:
                one_hand_data += line

    random.shuffle(data_all)
    num_all = len(data_all)
    n1 = int(num_all * 0.7)
    n2 = int(num_all * 0.85)
    data_train = data_all[:n1]
    data_valid = data_all[n1:n2]
    data_test = data_all[n2:]
    for hand_id, one_hand_data in data_train:
        target_path = os.path.join(train_dir, hand_id) + ".txt"
        with open(target_path, "w") as wf:
            wf.write(one_hand_data)
        print(f"train: {hand_id} done")
    for hand_id, one_hand_data in data_valid:
        target_path = os.path.join(valid_dir, hand_id) + ".txt"
        with open(target_path, "w") as wf:
            wf.write(one_hand_data)
        print(f"valid: {hand_id} done")
    for hand_id, one_hand_data in data_test:
        target_path = os.path.join(test_dir, hand_id) + ".txt"
        with open(target_path, "w") as wf:
            wf.write(one_hand_data)
        print(f"test: {hand_id} done")


class TileType(Enum):
    TONG = auto()    # 筒子
    BAMBOO = auto()  # 条子
    WAN = auto()     # 万子
    FENG = auto()    # 风牌
    JIAN = auto()    # 箭牌

    @property
    def prefix(self):
        # 获取对应的字母前缀
        return {
            TileType.TONG: 'T',
            TileType.BAMBOO: 'B',
            TileType.WAN: 'W',
            TileType.FENG: 'F',
            TileType.JIAN: 'J'
        }[self]


def tile_str2int(tile_str):
    prefix_map = {
        'T': TileType.TONG,
        'B': TileType.BAMBOO,
        'W': TileType.WAN,
        'F': TileType.FENG,
        'J': TileType.JIAN
    }

    F, S = tile_str
    tile_type = prefix_map[F]

    base = {
        TileType.TONG: 0,
        TileType.BAMBOO: 9,
        TileType.WAN: 18,
        TileType.FENG: 27,
        TileType.JIAN: 31
    }[tile_type]

    return base + (int(S) - 1)


def fans_str2dict(fans_str):
    result = {}
    fans = fans_str.split("+")
    for fan in fans:
        fan_type, fan_num = fan.split("*")
        result[fan_type] = fan_num
    return result
