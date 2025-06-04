import os
import argparse
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score

from pymj.agent.chinese_official_mahjiong.DQNAgent import Network
from pymj.botzone.SLDataset2 import SLDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default="Play")
    parser.add_argument('--dim_rnn', '-dr', type=int, default=64)
    parser.add_argument('--dropout', '-d', type=float, default=0.1)
    parser.add_argument('--pretrain_path', '-p', type=str, default="")
    parser.add_argument('--max_epochs', '-me', type=int, default=20)
    parser.add_argument('--add_noise', '-an', type=int, default=0)
    parser.add_argument('--use_weight', '-uw', type=int, default=1)
    args = parser.parse_args()
    assert args.mode in ["Play", "Chi", "Peng", "Gang"]

    model_dir = f"/root/autodl-tmp/mah_jiong/models{'' if args.noise else '_no'}_noise"
    train_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_hybrid_data/train/{args.mode}.pt", DEVICE)
    valid_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_hybrid_data/test/{args.mode}.pt", DEVICE)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1024)

    if os.path.exists(args.pretrain_path):
        pretrain_path = args.pretrain_path
    else:
        pretrain_path = None
    if args.mode == "Play":
        model = Network(34, 288, args.dim_rnn, args.dropout, pretrain_path)
    elif args.mode == "Gang":
        model = Network(4, 288, args.dim_rnn, args.dropout, pretrain_path)
    else:
        model = Network(1, 288, args.dim_rnn, args.dropout, pretrain_path)
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
                weighted_loss = (loss * weights).mean()
            else:
                weighted_loss = loss.mean()
            losses.append(loss.detach().item())
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()

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
