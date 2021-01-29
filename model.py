import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn.init


class CNN(nn.Module):  
    def __init__(self, classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2, padding=0)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, padding=0)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, padding=0)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2, 1, padding=0)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 5)
        self.active = nn.ReLU()

    def forward(self, x):
        x = self.active(self.conv1_bn(self.conv1(x)))
        x = self.active(self.conv2_bn(self.conv2(x)))
        x = self.active(self.conv3_bn(self.conv3(x)))
        x = self.active(self.conv4_bn(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
