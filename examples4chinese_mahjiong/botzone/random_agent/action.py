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
def choose_card2play(state):
    self_hand_card_ids = state["self_hand_card_ids"]
    return random.choice(self_hand_card_ids)


def choose_card2chi(state, middle_card_ids):
    return random.choice(middle_card_ids)


def choose_card2an_gang(state, gang_card_ids):
    return random.choice(gang_card_ids)


def choose_card2bu_gang(state, gang_card_ids):
    return random.choice(gang_card_ids)
# ================================================================================================
