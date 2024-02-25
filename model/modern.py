import torch
from torch import nn
import torch.nn.functional as F

class LeNet5Modern(nn.Module):
    def __init__(self, in_channels: int, feature_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=5)
        self.fc1 = nn.Linear(4*4*feature_channels, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x