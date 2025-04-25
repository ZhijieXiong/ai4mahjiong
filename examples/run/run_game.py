import numpy as np


from pymj.game.chinese_ofiicial_mahjiong.Game import Game
from pymj.agent.chinese_official_mahjiong.NoFuLuRandomAgent import NoFuLuRandomAgent
from pymj.agent.chinese_official_mahjiong.FuLuRandomAgent import FuLuRandomAgent


def test_game():
    game_wind = 0
    players_self_wind = list(range(4))
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    agents = [NoFuLuRandomAgent(random_generator) for _ in range(4)]
    game = Game(agents, game_wind, players_self_wind, random_generator=random_generator)
    initial_card_walls = Game.get_initial_card_walls(random_generator)
    game.init_game(initial_card_walls)
    game.run()
    game.print_game()


def test_many_games():
    game_wind = 0
    players_self_wind = list(range(4))
    num_game = 100
    num_hu = 0
    for random_seed in range(num_game):
        random_generator: np.random.RandomState = np.random.RandomState(random_seed)
        agents = [FuLuRandomAgent(random_generator) for _ in range(4)]
        game = Game(agents, game_wind, players_self_wind, random_generator=random_generator)
        initial_card_walls = Game.get_initial_card_walls(random_generator)
        game.init_game(initial_card_walls)
        game.run()
        if len(game.game_result[0]) == 1:
            num_hu += 1
    print(f"胡牌率：{num_hu / num_game * 100} %")
    

if __name__ == "__main__":
    test_game()
    test_many_games()