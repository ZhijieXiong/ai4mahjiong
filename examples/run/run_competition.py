import torch
import numpy as np


from pymj.game.chinese_ofiicial_mahjiong.Competition import Competition
from pymj.agent.chinese_official_mahjiong.FuLuRandomAgent import FuLuRandomAgent
from pymj.agent.chinese_official_mahjiong.EfficientAgent import EfficientAgent
from pymj.agent.chinese_official_mahjiong.SLBasedAgent import SLBasedAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_competition1():
    """
    副露消融实验，基于随机打牌模型
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            FuLuRandomAgent(random_generator),
            FuLuRandomAgent(random_generator, peng=False, gang=False),
            FuLuRandomAgent(random_generator, chi=False, gang=False),
            FuLuRandomAgent(random_generator, chi=False, peng=False)
        ], 256, random_generator=random_generator).run()
    

def test_competition2():
    """
    副露消融实验，基于牌效率打牌模型
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            EfficientAgent(random_generator),
            EfficientAgent(random_generator, peng=False, gang=False),
            EfficientAgent(random_generator, chi=False, gang=False),
            EfficientAgent(random_generator, chi=False, peng=False)
        ], 256, random_generator=random_generator).run()
    
    
def test_competition3():
    """
    吃牌选择消融实验，基于牌效率模型
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            EfficientAgent(random_generator, use_choose_card2_chi=True),
            EfficientAgent(random_generator, use_choose_card2_chi=False),
            EfficientAgent(random_generator, peng=False, gang=False, use_choose_card2_chi=True),
            EfficientAgent(random_generator, peng=False, gang=False, use_choose_card2_chi=False)
        ], 256, random_generator=random_generator).run()


def test_competition4():
    """
    对比牌效率模型和随机打牌模型
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            EfficientAgent(random_generator),
            EfficientAgent(random_generator),
            FuLuRandomAgent(random_generator),
            FuLuRandomAgent(random_generator)
        ], 256, random_generator=random_generator).run()


def test_competition5():
    """
    副露消融实验，基于SL-based模型
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_20_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_15_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_15_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_10_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_10_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_10_0.0001_1024_1e-06.ckt",
                device=DEVICE
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_20_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_15_0.0001_1024_1e-06.ckt",
                # peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_15_0.0001_1024_1e-06.ckt",
                # gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_10_0.0001_1024_1e-06.ckt",
                # an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_10_0.0001_1024_1e-06.ckt",
                # bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_10_0.0001_1024_1e-06.ckt",
                device=DEVICE
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_20_0.0001_256_1e-06.ckt",
                # chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_15_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_15_0.0001_1024_1e-06.ckt",
                # gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_10_0.0001_1024_1e-06.ckt",
                # an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_10_0.0001_1024_1e-06.ckt",
                # bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_10_0.0001_1024_1e-06.ckt",
                device=DEVICE
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_20_0.0001_256_1e-06.ckt",
                # chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_15_0.0001_1024_1e-06.ckt",
                # peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_15_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_10_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_10_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_10_0.0001_1024_1e-06.ckt",
                device=DEVICE
            )
        ], 256, random_generator=random_generator).run()
     

def test_competition6():
    """
    对比牌效率模型和SL-based模型，同时做吃牌选择消融实验
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_20_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_15_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_15_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_10_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_10_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_10_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=True
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_20_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_15_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_15_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_10_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_10_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_10_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=False
            ),
            EfficientAgent(random_generator, use_choose_card2_chi=True),
            EfficientAgent(random_generator, use_choose_card2_chi=False)
        ], 256, random_generator=random_generator).run()
    

if __name__ == "__main__":
    test_competition1()
    test_competition2()
    test_competition3()
    test_competition4()
    test_competition5()
    test_competition6()