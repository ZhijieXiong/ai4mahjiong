import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from pymj.agent.chinese_official_mahjiong.DQNAgent import Network, DeepNetwork
from pymj.botzone.SLDataset2 import SLDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


def evaluate_model(
    model_path,
    dim_rnn,
    use_deep,
):
    model_file_name: str = os.path.basename(model_path)
    mode = model_file_name.split("-")[0]
    test_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_hybrid_data/test/{mode}.pt", DEVICE)
    test_loader = DataLoader(test_set, batch_size=1024)
    if use_deep:
        Model = DeepNetwork
    else:
        Model = Network
    if mode == "Play":
        model = Model(34, 288, dim_rnn)
    elif mode == "Gang":
        model = Model(4, 288, dim_rnn)
    else:
        model = Model(1, 288, dim_rnn)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location="cpu")["state_dict"]
    )
    model.to(DEVICE)
    model.eval()

    all_pre_scores = []
    all_pre_labels = []
    all_labels = []
    total_samples = 0
    correct = 0
    for mlp_features_, cnn_features_, rnn_features_, rnn_seqs_len_, labels_, weights_, mask_ in test_loader:
        if mode == "Gang":
            output_ = model(mlp_features_, cnn_features_, rnn_features_, rnn_seqs_len_, mask_)
        else:
            output_ = model(mlp_features_, cnn_features_, rnn_features_, rnn_seqs_len_)
        if mode in ["Play", "Gang"]:
            pre_labels = output_.softmax(1).argmax(1)  # 获取预测类别
            batch_size = labels_.size(0)
            total_samples += batch_size
            correct += (pre_labels == labels_).sum().item()
            all_pre_labels.append(pre_labels.cpu().numpy())
        else:
            pre_scores = output_.squeeze(dim=-1)
            all_pre_scores.append(pre_scores.cpu().detach().numpy())
        all_labels.append(labels_.cpu().detach().numpy())

    if mode in ["Play", "Gang"]:
        acc = correct / total_samples
        print(f"{model_file_name.replace('.ckt', "")}, ACC: {acc}")
    else:
        auc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_pre_scores))
        print(f"{model_file_name.replace('.ckt', "")}, AUC: {auc}")



if __name__ == "__main__":
    dim_rnn = 64
    model_dir = "/root/autodl-tmp/mah_jiong/deep_models_no_weight_no_noise_no_pretrain_aug"
    use_deep = "deep_models" in model_dir
    model_names = [f"Play-{i}.ckt" for i in range(40,52)]
    model_paths = [os.path.join(model_dir, model_name) for model_name in model_names]
    for model_path in model_paths:
        evaluate_model(model_path, dim_rnn, use_deep)
    