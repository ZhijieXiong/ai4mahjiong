import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


@torch.no_grad()
def model_test(model, data_loader, mode):
    all_pred_scores = []
    all_pred_labels = []
    all_labels = []
    total_samples = 0
    correct = 0
    model.eval()
    for features, labels in data_loader:
        logits = model(features)
        if mode == "Play":
            pred_labels = logits.softmax(1).argmax(1)  # 获取预测类别
        else:
            pred_scores = logits.squeeze(dim=-1)
            all_pred_scores.append(pred_scores.cpu().numpy())
            pred_labels = torch.tensor([1 if p >= 0.5 else 0 for p in pred_scores]).to(labels.device)            

        # 更新统计
        batch_size = labels.size(0)
        total_samples += batch_size
        correct += (pred_labels == labels).sum().item()

        # 收集所有预测和标签（用于后续计算其他指标）
        all_pred_labels.append(pred_labels.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    # 计算整体准确率
    accuracy = correct / total_samples

    # 合并所有批次的预测结果
    all_pred_labels = np.concatenate(all_pred_labels)
    all_labels = np.concatenate(all_labels)

    # 计算分类报告（precision, recall, f1等）
    cls_report = classification_report(all_labels, all_pred_labels, output_dict=True, zero_division=0)

    # 返回所有指标
    results = {
        'accuracy': accuracy,
        'classification_report': cls_report
    }
    if mode != "Play":
        all_pred_scores = np.concatenate(all_pred_scores)
        results["auc"] = roc_auc_score(all_labels, all_pred_labels)

    return results
