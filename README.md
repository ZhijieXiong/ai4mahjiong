# 介绍
本项目旨在构建各类麻将游戏环境，以用于训练和测试麻将游戏智能体，同时对接[botzone](https://botzone.org.cn/)上的麻将游戏和bot

项目目前实现内容如下

（1）通用：国标麻将游戏环境模拟、国标麻将游戏智能体抽象

（2）botzone国标麻将游戏：解析botzone提供的[国标麻将强AI对局比赛数据](https://disk.pku.edu.cn/link/AA8CB7A57AFDCD48CAA7C749E04B5B6FAA)；实现botzone平台要求的国标麻将游戏智能体bot代码

（3）麻将游戏智能体：目前已实现随机打牌的麻将智能体、基于简单牌效率的麻将智能体、使用监督学习训练的基于CNN的麻将智能体


# 安装
由于项目还未开发完成，目前只支持从本地安装
```shell
git clone git@github.com:ZhijieXiong/ai4mahjiong.git && cd ai4mahjiong
pip install -e .
```

# 快速开始
## 模拟麻将游戏
运行`examples/run/run_game.py`可以查看使用随机打牌的麻将智能体进行单局和多局比赛的结果

运行`examples/run/run_competition.py`可以查看各类智能在[复式赛制比赛](https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong)下的得分对比

## 开发自己的智能体
需要实现一个继承`pymj/agent/chinese_official_mahjiong/Agent.py`的类，并实现其中的抽象方法，可参考`pymj/agent/chinese_official_mahjiong/NoFuLuRandomAgent.py`

## 训练基于神经网络的智能体
首先运行`examples/dnn/get_game_data.py`获得解析后的对局数据，然后划分数据集。本项目是将一局比赛视为一个“样本”，需要将训练集比赛数据、验证集比赛数据和测试集比赛数据分别放在三个目录（分别命名`train`、`valid`和`test`）下，并三个文件夹放在同一目录下

然后运行`examples/dnn/get_dnn_data.py`获得不同动作的数据。本项目将作出动作之前的状态视为样本特征，将动作视为标签。本项目实现了6个动作的数据处理和模型训练，分别是 Play(34) Chi(2) Peng(2) Gang(2) AnGang(2) BuGang(2) ，其中Play是多分类模型，表示选择要打出的牌，其余均为二分类模型，即选择是否做这个动作

最后运行`examples/dnn/train.py`以训练不同的模型

# 分析和对局结果解释
`analysis_result.txt`是对botzone数据的分析结果

`run_competition_result.txt`是`run_competition.py`的运行结果

# 注意事项
本项目的麻将游戏模拟过程和botzone的麻将游戏模拟过程有以下区别：
（1）botzone设置当每个人的牌墙摸完时，一局比赛结束；本项目则是当所有牌摸完时，才结束比赛。当比赛过程中没有选手杠牌时，二者完全一致
（2）本项目实现的复试赛制与botzone有细微差别