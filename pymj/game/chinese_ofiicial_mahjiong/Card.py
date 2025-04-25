class Card:
    @staticmethod
    def encoding(card_str: str) -> int:
        assert len(card_str) == 2
        F: str = card_str[0]
        S: str = card_str[1]
        assert 65 <= ord(F) <= 90
        assert 48 <= ord(S) <= 57
        if F == "W":
            return int(S) - 1
        elif F == "T":
            return 8 + int(S)
        elif F == "B":
            return 17 + int(S)
        elif F == "F":
            return 26 + int(S)
        else:
            return 30 + int(S)

    @staticmethod
    def decoding(card_id: int) -> str:
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
