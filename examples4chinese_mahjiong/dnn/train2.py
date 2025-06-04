import os
import argparse
import torch
import random
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score

from pymj.agent.chinese_official_mahjiong.DQNAgent import Network, DeepNetwork
from pymj.botzone.SLDataset2 import SLDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mirror_edge_tile(hand_counts, label):
    """
    将边缘牌进行镜像增强，例如1万变成9万，仅对孤张进行处理。
    """
    mirrors = {
        0: 8, 1: 7, 2: 6, 6: 2, 7: 1, 8: 0,       # 万
        9: 17, 10: 16, 11: 15, 15: 11, 16: 10, 17: 9,  # 条
        18: 26, 19: 25, 20: 24, 24: 20, 25: 19, 26: 18  # 筒
    }

    for idx in list(mirrors.keys()):
        if hand_counts[idx] == 1:
            # 仅增强孤张，且无前后搭子
            prev = idx - 1 if idx % 9 != 0 else -1
            next = idx + 1 if idx % 9 != 8 else -1
            if ((prev == -1 or hand_counts[prev] == 0) and 
                (next == -1 or hand_counts[next] == 0)):
                mirror_idx = mirrors[idx]
                if hand_counts[mirror_idx] == 0:
                    hand_counts[idx] -= 1
                    hand_counts[mirror_idx] += 1
                    if label == idx:
                        label = mirror_idx
                    break  # 一次只做一个替换
    return hand_counts, label


def random_drop_tile(hand_counts, label):
    """
    随机移除一张非label牌并加入一张合法新牌。
    """
    candidates = (hand_counts > 0).nonzero(as_tuple=True)[0].tolist()
    if label in candidates:
        candidates.remove(label)
    if not candidates:
        return hand_counts, label
    drop = random.choice(candidates)
    hand_counts[drop] -= 1
    # 随机加一个不超过4张的牌
    add_candidates = [i for i in range(34) if hand_counts[i] < 4]
    if add_candidates:
        add = random.choice(add_candidates)
        hand_counts[add] += 1
    return hand_counts, label


def shift_tile(hand_counts, label):
    """
    将某个顺子相关牌左右平移（如2→3，7→6）
    """
    shift_candidates = []
    for i in range(34):
        if 0 < i < 33 and hand_counts[i] > 0:
            if i % 9 != 0 and i % 9 != 8:
                shift_candidates.append(i)
    if not shift_candidates:
        return hand_counts, label
    i = random.choice(shift_candidates)
    shift = random.choice([-1, 1])
    j = i + shift
    if 0 <= j < 34 and hand_counts[j] < 4:
        hand_counts[i] -= 1
        hand_counts[j] += 1
        if label == i:
            label = j
    return hand_counts, label


def augment_batch(mlp_features, cnn_features, labels, enable_label_aug=True):
    """
    增强一个batch的样本，仅修改手牌相关特征，包含：
    - mirror_edge_tile
    - random_drop_tile
    - shift_tile

    输入：
    - mlp_features: Tensor (batch, 288)
    - cnn_features: Tensor (batch, 4, 34)
    - labels: Tensor (batch,)
    输出：
    - 新的mlp_features, cnn_features, labels（增强后的）
    """
    batch_size = mlp_features.size(0)
    mlp_features = mlp_features.clone()
    cnn_features = cnn_features.clone()
    labels = labels.clone()

    for i in range(batch_size):
        # 手牌计数（shape: 34）
        hand_counts = mlp_features[i, 8:42].int().clone()

        # 选择1~2种策略组合增强
        strategies = [mirror_edge_tile, random_drop_tile, shift_tile]
        chosen = random.sample(strategies, 2)

        label = labels[i].item()
        for strat in chosen:
            hand_counts, label = strat(hand_counts, label if enable_label_aug else labels[i].item())

        # 更新 cnn 特征
        cnn_features[i].zero_()
        for j in range(34):
            for k in range(hand_counts[j]):
                cnn_features[i, k, j] = 1

        # 更新 mlp 特征中第9~42维手牌计数（共34维）
        mlp_features[i, 8:42] = hand_counts.float()

        # 更新标签
        if args.mode == "Play":
            if enable_label_aug and (type(label) is int) and (not (0 <= label < 34)):
                labels[i] = label

    return mlp_features, cnn_features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default="Play")
    parser.add_argument('--dim_rnn', '-dr', type=int, default=64)
    parser.add_argument('--dropout', '-d', type=float, default=0.1)
    parser.add_argument('--pretrain_path', '-p', type=str, default="")
    parser.add_argument('--max_epochs', '-me', type=int, default=20)
    parser.add_argument('--add_noise', '-an', type=int, default=0)
    parser.add_argument('--use_weight', '-uw', type=int, default=0)
    parser.add_argument('--use_deep', '-ud', type=int, default=1)
    parser.add_argument('--use_aug', '-ua', type=int, default=1)
    args = parser.parse_args()
    assert args.mode in ["Play", "Chi", "Peng", "Gang"]

    if os.path.exists(args.pretrain_path):
        pretrain_path = args.pretrain_path
        use_pretrain = True
    else:
        pretrain_path = None
        use_pretrain = False
    use_deep = args.use_deep
    model_dir = f"/root/autodl-tmp/mah_jiong/{'deep_' if use_deep else ''}models{'' if args.use_weight else '_no'}_weight{'' if args.add_noise else '_no'}_noise{'' if use_pretrain else '_no'}_pretrain{'_aug' if args.use_aug else ''}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    train_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_hybrid_data/train/{args.mode}.pt", DEVICE)
    valid_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_hybrid_data/test/{args.mode}.pt", DEVICE)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1024)
    if use_deep:
        Model = DeepNetwork
    else:
        Model = Network
    if args.mode == "Play":
        model = Model(34, 288, args.dim_rnn, args.dropout, pretrain_path)
    elif args.mode == "Gang":
        model = Model(4, 288, args.dim_rnn, args.dropout, pretrain_path)
    else:
        model = Model(1, 288, args.dim_rnn, args.dropout, pretrain_path)
    model.to(DEVICE)

    # 定义参数分组
    param_groups = [
        # MLP分支
        {
            'params': model.mlp_branch.parameters(),
            'lr': 1e-3,
            'weight_decay': 0.02
        },
        # CNN分支
        {
            'params': model.cnn_branch.parameters(),
            'lr': 5e-4,
            'weight_decay': 0.005
        },
        # RNN分支（包含embedding和GRU）
        {
            'params': list(model.embed_played_card.parameters()) +
                    list(model.rnn.parameters()) +
                    list(model.rnn_fc.parameters()),
            'lr': 2e-4,
            'weight_decay': 0.001
        },
        # 融合层
        {
            'params': model.fusion.parameters(),
            'lr': 8e-4,
            'weight_decay': 0.01
        }
    ]
    optim = AdamW(param_groups, betas=(0.9, 0.99), eps=1e-8)

    # 分阶段调度器组合
    total_samples = len(train_set)
    batch_size = train_loader.batch_size
    steps_per_epoch = (total_samples + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * args.max_epochs
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, end_factor=1.0,
                                              total_iters=int(0.1 * total_steps)),
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(0.8 * total_steps),
                                                       eta_min=1e-5),  # 使用固定最小值
            torch.optim.lr_scheduler.ConstantLR(optim, factor=1.0,
                                                total_iters=int(0.1 * total_steps))
        ],
        milestones=[
            int(0.2 * total_steps),
            int(0.8 * total_steps)
        ]
    )

    if args.mode in ["Play", "Gang"]:
        loss_fcn = CrossEntropyLoss(reduction='none')
    else:
        loss_fcn = BCEWithLogitsLoss(reduction='none')

    current_step = 0
    step2save = total_samples // batch_size
    if args.mode == "Play":
        step2save = (total_samples // batch_size) // 3
    elif args.mode == "Chi":
        step2save = (total_samples // batch_size) // 2
    else:
        step2save = total_samples // batch_size
    max_acc = 0
    max_auc = 0
    best_step_num = 0
    losses = []
    add_noise = args.add_noise == 1
    for epoch in range(args.max_epochs):
        for mlp_features, cnn_features, rnn_features, rnn_seqs_len, labels, weights, mask in train_loader:
            if args.mode == "Gang":
                output = model(mlp_features, cnn_features, rnn_features, rnn_seqs_len, mask, add_noise)
            else:
                output = model(mlp_features, cnn_features, rnn_features, rnn_seqs_len, noise=add_noise)
            if args.mode in ["Play", "Gang"]:
                loss = loss_fcn(output, labels)
            else:
                loss = loss_fcn(output.squeeze(dim=-1), labels.float())
            if args.use_weight:
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()
            losses.append(loss.detach().item())
            loss.backward()
            optim.step()
            optim.zero_grad()
            if args.use_aug:
                auged_mlp_features, auged_cnn_features, auged_labels = augment_batch(mlp_features, cnn_features, labels)
                if args.mode == "Gang":
                    output = model(auged_mlp_features.to(DEVICE), auged_cnn_features.to(DEVICE), rnn_features, rnn_seqs_len, mask, add_noise)
                else:
                    output = model(auged_mlp_features.to(DEVICE), auged_cnn_features.to(DEVICE), rnn_features, rnn_seqs_len, noise=add_noise)
                if args.mode in ["Play", "Gang"]:
                    auged_loss = loss_fcn(output, auged_labels.to(DEVICE))
                else:
                    auged_loss = loss_fcn(output.squeeze(dim=-1), auged_labels.float().to(DEVICE))
                if args.use_weight:
                    auged_loss = (auged_loss * weights).mean()
                else:
                    auged_loss = auged_loss.mean()
                losses.append(auged_loss.detach().item())
                auged_loss.backward()
                optim.step()
                optim.zero_grad()
            scheduler.step()

            current_step += 1
            if current_step % step2save == 0:
                model_path = os.path.join(model_dir, f"{args.mode}-{current_step // step2save}.ckt")
                all_pre_scores = []
                all_pre_labels = []
                all_labels = []
                total_samples = 0
                correct = 0
                model.eval()
                for mlp_features_, cnn_features_, rnn_features_, rnn_seqs_len_, labels_, weights_, mask_ in valid_loader:
                    if args.mode == "Gang":
                        output_ = model(mlp_features_, cnn_features_, rnn_features_, rnn_seqs_len_, mask_)
                    else:
                        output_ = model(mlp_features_, cnn_features_, rnn_features_, rnn_seqs_len_)
                    if args.mode in ["Play", "Gang"]:
                        pre_labels = output_.softmax(1).argmax(1)  # 获取预测类别
                        batch_size = labels_.size(0)
                        total_samples += batch_size
                        correct += (pre_labels == labels_).sum().item()
                        all_pre_labels.append(pre_labels.cpu().numpy())
                    else:
                        pre_scores = output_.squeeze(dim=-1)
                        all_pre_scores.append(pre_scores.cpu().detach().numpy())
                    all_labels.append(labels_.cpu().detach().numpy())

                ave_loss = sum(losses) / len(losses)
                if args.mode in ["Play", "Gang"]:
                    acc = correct / total_samples
                    if acc > max_acc:
                        max_acc = acc
                        best_step_num = current_step // step2save
                    print(f"step num {current_step // step2save} ACC: {acc}, loss: {ave_loss}")
                else:
                    auc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_pre_scores))
                    if auc > max_auc:
                        max_auc = auc
                        best_step_num = current_step // step2save
                    print(f"step num {current_step // step2save} AUC: {auc}, loss: {ave_loss}")
                torch.save({"state_dict": model.state_dict()}, model_path)
                losses = []
                model.train()

    if args.mode in ["Play", "Gang"]:
        print(f"best ACC: {max_acc}, best step num: {best_step_num}")
    else:
        print(f"best AUC: {max_auc}, best step num: {best_step_num}")
