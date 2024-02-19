import torch
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
        
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s2(self.tanh(self.c1(x)))
        x = self.s4(self.tanh(self.c3(x)))
        x = self.flatten(self.c5(x))
        x = self.output(self.tanh(self.f6(x)))
        return x

class LeNet5Functional(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s2(F.tanh(self.c1(x)))
        x = self.s4(F.tanh(self.c3(x)))
        x = torch.flatten(self.c5(x), 1)
        x = self.output(F.tanh(self.f6(x)))
        return x

class LeNet5Sequential(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass    

class LeNet5Modern(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass    