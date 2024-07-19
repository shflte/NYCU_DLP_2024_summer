# implement SCCNet model

import torch
import torch.nn as nn

# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super(SquareLayer, self).__init__()

    def forward(self, x):
        return x ** 2

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=1000, Nu=22, C=22, Nc=44, Nt=1, dropoutRate=0.5):
        super(SCCNet, self).__init__()

        self.firstConvBlock = nn.Sequential(
            nn.Conv2d(1, Nu, (C, Nt), padding=0),
            nn.BatchNorm2d(Nu),
        )

        self.secondConvBlock = nn.Sequential(
            nn.Conv2d(1, Nc, (Nu, 12), padding=(0, 6)),
            nn.BatchNorm2d(Nc),
        )

        self.squareLayer = SquareLayer()

        self.pool = nn.AvgPool2d((1, 62))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(Nc * (timeSample // 62), numClasses)
        self.dropout = nn.Dropout(dropoutRate)
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        x = self.firstConvBlock(x)

        # Permute dimensions
        x = x.permute(0, 2, 1, 3)

        x = self.secondConvBlock(x)

        x = self.squareLayer(x)

        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.softmax(x)

        return x

    def get_size(self, C, N):
        pass
