import os
import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score

from pymj.agent.chinese_official_mahjiong.DQNAgent import Network
from pymj.botzone.SLDataset2 import SLDataset
from pymj.dnn_model.utils import model_test


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default="Gang")
    parser.add_argument('--dim_rnn', '-dr', type=int, default=64)
    parser.add_argument('--dropout', '-d', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.000001)
    args = parser.parse_args()
    assert args.mode in ["Play", "Chi", "Peng", "Gang", "AnGang", "BuGang"], \
        f'mode must be one of ["Play", "Chi", "Peng", "Gang", "AnGang", "BuGang"]'

    model_dir = "/root/autodl-tmp/mah_jiong/models"
    train_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_hybrid_data/train/{args.mode}.pt", DEVICE)
    valid_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_hybrid_data/test/{args.mode}.pt", DEVICE)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=4096)

    if args.mode == "Play":
        model = Network(34, 288, args.dim_rnn, args.dropout)
    elif args.mode == "Gang":
        model = Network(4, 288, args.dim_rnn, args.dropout)
    else:
        model = Network(1, 288, args.dim_rnn, args.dropout)
    model.to(DEVICE)
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', patience=1)
    if args.mode in ["Play", "Gang"]:
        loss_fcn = CrossEntropyLoss()
    else:
        loss_fcn = BCEWithLogitsLoss()
    epochs = args.epochs

    current_step = 0
    if args.mode == "Play":
        step2save = 1000
    elif args.mode == "Chi":
        step2save = 400
    elif args.mode == "Peng":
        step2save = 200
    else:
        step2save = 100
    max_acc = 0
    max_auc = 0
    best_step = 0
    losses = []
    for _ in range(epochs):
        for features, labels in train_loader:
            output = model(features)
            if args.mode == "Play":
                loss = loss_fcn(output, labels)
            else:
                loss = loss_fcn(output.squeeze(dim=-1), labels.float())
            losses.append(loss.detach().item())
            optim.zero_grad()
            loss.backward()
            optim.step()

            current_step += 1
            if current_step % step2save == 0:
                if current_step > 0:
                    model_path = os.path.join(model_dir, f"{args.mode}-{current_step // step2save}.ckt")
                    torch.save({"state_dict": model.state_dict()}, model_path)
                model.eval()

                all_pre_scores = []
                all_pre_labels = []
                all_labels = []
                total_samples = 0
                correct = 0
                model.eval()
                for features_, labels_ in valid_loader:
                    logits_ = model(features_)
                    if args.mode in ["Play", "Gang"]:
                        pre_labels = logits_.softmax(1).argmax(1)  # 获取预测类别
                    else:
                        pre_scores = logits_.squeeze(dim=-1)
                        all_pre_scores.append(pre_scores.cpu().numpy())
                        pre_labels = torch.tensor([1 if p >= 0.5 else 0 for p in pre_scores]).to(labels.device)
                    batch_size = labels.size(0)
                    total_samples += batch_size
                    correct += (pre_labels == labels).sum().item()
                    all_pre_labels.append(pre_labels.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

                ave_loss = sum(losses) / len(losses)
                if args.mode in ["Play", "Gang"]:
                    acc = correct / total_samples
                    if acc > max_acc:
                        max_acc = acc
                        best_step = current_step
                    print(f"step num {current_step // step2save} ACC: {acc}, loss: {ave_loss}")
                else:
                    auc = roc_auc_score(all_labels, all_pre_labels)
                    if auc > max_auc:
                        max_auc = auc
                        best_step = current_step
                    print(f"step num {current_step // step2save} AUC: {auc}, loss: {ave_loss}")
                losses = []
                model.train()

    if args.mode in ["Play", "Gang"]:
        print(f"best ACC: {max_acc}, best step num: {best_step // step2save}")
    else:
        print(f"best AUC: {max_auc}, best step num: {best_step // step2save}")
