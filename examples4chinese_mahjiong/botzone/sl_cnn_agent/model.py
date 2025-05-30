import os
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return x + self.layers(x)


class PlayModel(nn.Module):
    def __init__(self, in_channels, num_layers=20):
        super(PlayModel, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        self.res_blocks = nn.Sequential(
            *(ResBlock() for _ in range(num_layers))
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.BatchNorm1d(1)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.res_blocks(x)
        x = self.out_conv(x)
        return x.squeeze(1)


class FuroModel(nn.Module):
    def __init__(self, in_channels, num_layers=20):
        super(FuroModel, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        self.res_blocks = nn.Sequential(
            *(ResBlock() for _ in range(num_layers))
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(256, 3, kernel_size=1),
            nn.BatchNorm1d(3),
            nn.LeakyReLU(0.2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 35, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.res_blocks(x)
        x = self.out_conv(x)
        x = self.fc(x)
        return x


def load_model(model_path, model_type, device):
    assert model_type in ["Play", "Furo"]
    model_file_name = os.path.basename(model_path)
    num_layer = int(model_file_name.split("_")[1])
    if model_type == "Play":
        model: PlayModel = PlayModel(in_channels=28, num_layers=num_layer)
    else:
        model: FuroModel = FuroModel(in_channels=28, num_layers=num_layer)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")["state_dict"]
    )
    return model.to(device)
