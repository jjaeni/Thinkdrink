import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchsummary import summary

class ResNet50V2CustomLayer(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50V2CustomLayer, self).__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V2")
        # layer 가중치 동결
        for param in resnet.parameters():
            param.requires_grad=False
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # insert layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048*4*4, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)

        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)

        x = F.log_softmax(self.fc3(x), dim=1)
        return x