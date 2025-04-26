import torch
import numpy as np


from pymj.game.chinese_ofiicial_mahjiong.Competition import Competition
from pymj.agent.chinese_official_mahjiong.FuLuRandomAgent import FuLuRandomAgent
from pymj.agent.chinese_official_mahjiong.EfficientAgent import EfficientAgent
from pymj.agent.chinese_official_mahjiong.SLBasedAgent import SLBasedAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_competition1():
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            FuLuRandomAgent(random_generator),
            FuLuRandomAgent(random_generator, peng=False, gang=False),
            FuLuRandomAgent(random_generator, chi=False, gang=False),
            FuLuRandomAgent(random_generator, chi=False, peng=False)
        ], 128, random_generator=random_generator).run()


def test_competition2():
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            EfficientAgent(random_generator),
            EfficientAgent(random_generator),
            FuLuRandomAgent(random_generator),
            FuLuRandomAgent(random_generator)
        ], 128, random_generator=random_generator).run()
    

def test_competition3():
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            SLBasedAgent(
                peng_model_path="/root/autodl-tmp/mah_jiong_models/Peng_7_0.0005_512_1e-05.pt",
                gang_model_path="/root/autodl-tmp/mah_jiong_models/Gang_7_0.0005_512_1e-05.pt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong_models/AnGang_7_0.0005_512_1e-05.pt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong_models/BuGang_7_0.0005_512_1e-05.pt",
                device=DEVICE
            ),
            EfficientAgent(random_generator),
            EfficientAgent(random_generator),
            FuLuRandomAgent(random_generator)
        ], 128, random_generator=random_generator).run()
    

if __name__ == "__main__":
    # test_competition1()
    # test_competition2()
    test_competition3()