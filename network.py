import torch
from torch.nn import Module

class NN_testMod(Module):
    def __init__(self) -> None:
        super().__init__()
        self.nn_model_1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(1, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 64),
            torch.nn.Linear(64, 10)
        )

    def forward(self, input):
        return self.nn_model_1(input)