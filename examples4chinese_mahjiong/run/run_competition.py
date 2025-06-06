import torch
import functools
import numpy as np

from pymj.game.chinese_ofiicial_mahjiong.Competition import Competition
from pymj.agent.chinese_official_mahjiong.FuLuRandomAgent import FuLuRandomAgent
from pymj.agent.chinese_official_mahjiong.EfficientAgent import EfficientAgent
from pymj.agent.chinese_official_mahjiong.SLBasedAgent import SLBasedAgent
from pymj.agent.chinese_official_mahjiong.SLBasedHybirdAgent import SLBasedAgent as SLBasedHybirdAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log_run(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        line = "=" * 100 + "\n" + " " * 40 + f"开始运行{func.__name__}"
        print(line)
        result = func(*args, **kwargs)
        print("=" * 100)
        return result
    return wrapper


@log_run
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


@log_run
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


@log_run
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


@log_run
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


@log_run
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
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                # peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                # gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                # an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                # bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                # chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                # gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                # an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                # bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                # chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                # peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE
            )
        ], 256, random_generator=random_generator).run()


@log_run
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
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=True
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=False
            ),
            EfficientAgent(random_generator, use_choose_card2_chi=True),
            EfficientAgent(random_generator, use_choose_card2_chi=False)
        ], 256, random_generator=random_generator).run()


@log_run
def test_competition7():
    """
    模型大小消融实验
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=True
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=True
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_7_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_5_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_3_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_3_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_3_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=True
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_7_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_5_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_3_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_3_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_3_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=True
            ),
        ], 256, random_generator=random_generator).run()


@log_run
def test_competition8():
    """
    对比混合模型使用加权损失和不使用加权损失
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models_no_weight_no_noise_no_pretrain/Chi-40.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models_no_weight_no_noise_no_pretrain/Peng-16.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models_no_weight_no_noise_no_pretrain/Gang-13.ckt",
                device=DEVICE
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models_no_weight_no_noise_no_pretrain/Play-44.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models_no_weight_no_noise_no_pretrain/Chi-36.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models_no_weight_no_noise_no_pretrain/Peng-12.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models_no_weight_no_noise_no_pretrain/Gang-10.ckt",
                device=DEVICE
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models_weight_no_noise_no_pretrain/Chi-40.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models_weight_no_noise_no_pretrain/Peng-14.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models_weight_no_noise_no_pretrain/Gang-12.ckt",
                device=DEVICE
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models_weight_no_noise_no_pretrain/Play-44.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models_weight_no_noise_no_pretrain/Chi-36.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models_weight_no_noise_no_pretrain/Peng-12.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models_weight_no_noise_no_pretrain/Gang-10.ckt",
                device=DEVICE
            ),
        ], 256, random_generator=random_generator).run()


@log_run
def test_competition9():
    """
    对比Deep混合模型使用加权损失和不使用加权损失
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-46.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-28.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-10.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-8.ckt",
                device=DEVICE,
                deep=True
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_weight_no_noise_no_pretrain/Chi-32.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_weight_no_noise_no_pretrain/Peng-12.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_weight_no_noise_no_pretrain/Gang-7.ckt",
                device=DEVICE,
                deep=True
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_weight_no_noise_no_pretrain/Play-46.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_weight_no_noise_no_pretrain/Gang-6.ckt",
                device=DEVICE,
                deep=True
            ),
        ], 256, random_generator=random_generator).run()


@log_run
def test_competition10():
    """
    对比不同程度的保守副露策略
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.5,
                n2=1.0
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.6,
                n2=1.0
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.7,
                n2=1.0
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.8,
                n2=1.0
            ),
        ], 256, random_generator=random_generator).run()


@log_run
def test_competition11():
    """
    对比不同程度的保守副露策略
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.5,
                n2=1.0
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.5,
                n2=0.95
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.5,
                n2=0.9
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.5,
                n2=0.85
            ),
        ], 256, random_generator=random_generator).run()


@log_run
def test_competition12():
    """
    纯CNN模型和混合模型对比
    """
    random_seed = 0
    random_generator: np.random.RandomState = np.random.RandomState(random_seed)
    Competition(
        [
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=True
            ),
            SLBasedAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/models/Play_10_0.0001_256_1e-06.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/models/Chi_10_0.0001_1024_1e-06.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/models/Peng_7_0.0001_1024_1e-06.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/models/Gang_5_0.0001_1024_1e-06.ckt",
                an_gang_model_path="/root/autodl-tmp/mah_jiong/models/AnGang_5_0.0001_1024_1e-06.ckt",
                bu_gang_model_path="/root/autodl-tmp/mah_jiong/models/BuGang_5_0.0001_1024_1e-06.ckt",
                device=DEVICE, use_choose_card2_chi=True
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.5,
                n2=1.0
            ),
            SLBasedHybirdAgent(
                random_generator,
                play_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain_aug/Play-48.ckt",
                chi_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Chi-30.ckt",
                peng_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Peng-11.ckt",
                gang_model_path="/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain/Gang-9.ckt",
                device=DEVICE,
                deep=True,
                n1=0.5,
                n2=1.0
            ),
        ], 256, random_generator=random_generator).run()


if __name__ == "__main__":
    # test_competition1()
    # test_competition2()
    # test_competition3()
    # test_competition4()
    # test_competition5()
    # test_competition6()
    # test_competition7()
    # test_competition8()
    # test_competition9()
    # test_competition10()
    # test_competition11()
    test_competition12()