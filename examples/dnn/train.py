import os
import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from pymj.dnn_model.resnet import *
from pymj.botzone.SLDataset1 import SLDataset
from pymj.dnn_model.utils import model_test


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default="AnGang")
    parser.add_argument('--num_layers', '-nl', type=int, default=5)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.000001)
    parser.add_argument('--batch_size', '-bs', type=int, default=1024)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    args = parser.parse_args()
    assert args.mode in ["Play", "Chi", "Peng", "Gang", "AnGang", "BuGang"], \
        f'mode must be one of ["Play", "Chi", "Peng", "Gang", "AnGang", "BuGang"]'

    model_dir = "/root/autodl-tmp/mah_jiong/models"
    best_model_path = os.path.join(model_dir, f"{args.mode}_{args.num_layers}_{args.learning_rate}_{args.batch_size}_{args.weight_decay}.ckt")
    train_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_data/train/{args.mode}.pt", DEVICE)
    valid_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_data/valid/{args.mode}.pt", DEVICE)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=2048)

    if args.mode == "Play":
        model = PlayModel(num_layers=args.num_layers, in_channels=28)
    else:
        model = FuroModel(num_layers=args.num_layers, in_channels=28)
    model.to(DEVICE)
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', patience=1)
    if args.mode == "Play":
        loss_fcn = CrossEntropyLoss()
    else:
        loss_fcn = BCEWithLogitsLoss()
    epochs = args.epochs

    max_acc = 0
    max_auc = 0
    losses = []
    for epoch in range(epochs):
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
        model.eval()
        evaluation_result = model_test(model, valid_loader, args.mode)
        acc = evaluation_result["accuracy"]
        ave_loss = sum(losses) / len(losses)
        if args.mode == "Play":
            print(f"epoch {epoch+1} ACC: {acc}, loss: {ave_loss}")
        else:
            print(f"epoch {epoch+1} AUC: {evaluation_result['auc']}, ACC: {acc}, loss: {ave_loss}")
        losses = []
        if args.mode == "Play":
            if acc > max_acc:
                max_acc = acc
                torch.save({"state_dict": model.state_dict()}, best_model_path)
        else:
            if evaluation_result["auc"] > max_auc:
                max_auc = evaluation_result["auc"]
                max_acc = acc
                torch.save({"state_dict": model.state_dict()}, best_model_path)
        model.train()
        scheduler.step(acc)
    if args.mode == "Play":
        print(f"best ACC: {max_acc}")
    else:
        print(f"best AUC: {max_auc}, ACC: {max_acc}")
