import os
import torch
from torch.utils.data import DataLoader

from pymj.dnn_model.resnet import *
from pymj.botzone.SLDataset1 import SLDataset
from pymj.dnn_model.utils import model_test


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/mah_jiong/models/Chi_15_0.0001_1024_1e-06.ckt"
    model_file_name: str = os.path.basename(model_path)
    mode: str = (model_file_name.split("_")[0])
    num_layer: int = int(model_file_name.split("_")[1])
    test_set = SLDataset(f"/root/autodl-tmp/mah_jiong/sl_data/test/{mode}.pt", DEVICE)
    test_loader = DataLoader(test_set, batch_size=4096, shuffle=False)

    if mode == "Play":
        model = PlayModel(num_layers=num_layer, in_channels=28)
    else:
        model = FuroModel(num_layers=num_layer, in_channels=28)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location="cpu")["state_dict"]
    )
    model.to(DEVICE)
    model.eval()
    evaluation_result = model_test(model, test_loader, mode)
    print(evaluation_result)
    