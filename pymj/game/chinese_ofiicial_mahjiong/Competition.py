import numpy as np
from copy import deepcopy

from pymj.game.chinese_ofiicial_mahjiong.Game import Game
from pymj.agent.chinese_official_mahjiong.Agent import Agent


def permute(n):
    nums = list(range(n))

    def backtrack(start, end):
        if start == end:
            result.append(nums.copy())  # 当前排列完成，加入结果
        for i in range(start, end):
            nums[start], nums[i] = nums[i], nums[start]  # 交换
            backtrack(start + 1, end)  # 递归下一层
            nums[start], nums[i] = nums[i], nums[start]  # 撤销交换（回溯）

    result = []
    backtrack(0, len(nums))
    return result


class Competition:
    """复试赛制"""
    def __init__(self, agents: list[Agent], num_round: int = 128, random_generator: np.random.RandomState = None, random_seed: int = 0):
        self.agents: list[Agent] = agents
        self.num_round: int = num_round
        self.players_game_score: list[int] = [0] * 4
        self.players_score: list[int] = [0] * 4
        if random_generator is None:
            self.random_generator: np.random.RandomState = np.random.RandomState(random_seed)
        else:
            self.random_generator: np.random.RandomState = random_generator
        self.all_permutation = permute(4)

    def run(self):
        for r in range(self.num_round):
            game_wind: int = self.random_generator.choice(list(range(4)))
            players_self_wind: list[int] = list(range(4))
            initial_card_walls: list[list[int]] = Game.get_initial_card_walls(self.random_generator)
            # print(f"Round {r+1}")
            for i in range(24):
                order: list = self.all_permutation[i]
                players_self_wind_: list = []
                initial_card_walls_: list[list[int]] = []
                for j in order:
                    players_self_wind_.append(players_self_wind[j])
                    initial_card_walls_.append(deepcopy(initial_card_walls[j]))
                game: Game = Game(self.agents, game_wind, players_self_wind_, random_generator=self.random_generator)
                game.init_game(initial_card_walls_)
                game.run()
                if len(game.game_result[0]) == 1:
                    last_player_id: int = game.last_observation[0]
                    win_player_id: int = game.game_result[0][0]
                    fan_score = 0
                    for s, t in game.game_result[1][0]:
                        fan_score += s
                    if last_player_id == win_player_id:
                        # 自摸：三家各扣（8+fan）
                        for k in range(4):
                            if k == win_player_id:
                                self.players_game_score[k] += (8 + fan_score) * 3
                            else:
                                self.players_game_score[k] -= (8 + fan_score)
                    else:
                        # 点炮：点炮者扣（8+fan）分数，剩余两家扣8分
                        for k in range(4):
                            if k == win_player_id:
                                self.players_game_score[k] += 24 + fan_score
                            elif k == last_player_id:
                                self.players_game_score[k] -= (8 + fan_score)
                            else:
                                self.players_game_score[k] -= 8
                # print(f"    Game Scores: {', '.join(list(map(str, self.players_game_score)))}")
            self.update_players_score()
            print(f"    Round Scores: {', '.join(list(map(str, self.players_score)))}")
            self.players_game_score = [0] * 4

    def update_players_score(self):
        # 将玩家按分数从小到大排序，并保留原始索引
        sorted_players = sorted(enumerate(self.players_game_score), key=lambda x: x[1])

        # 初始化每个玩家的加分
        score_to_add = [0] * 4

        current_rank = 1  # 当前分组的分数（从1开始）
        i = 0
        n = 4

        while i < n:
            current_score = sorted_players[i][1]

            # 找到所有与当前分数相同的玩家
            j = i
            while j < n and sorted_players[j][1] == current_score:
                j += 1

            # 为这一组的玩家分配 current_rank
            for k in range(i, j):
                player_id = sorted_players[k][0]
                score_to_add[player_id] = current_rank

            # 下一分组分数加1，跳到下一组
            current_rank += 1
            i = j

        # 更新总分数
        for player_id in range(4):
            self.players_score[player_id] += score_to_add[player_id]
