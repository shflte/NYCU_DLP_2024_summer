# implement SCCNet model

import torch
import torch.nn as nn

# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass

class SCCNet(nn.Module):
    def __init__(self, numClasses=0, timeSample=0, Nu=0, C=0, Nc=0, Nt=0, dropoutRate=0):
        super(SCCNet, self).__init__()
        pass

    def forward(self, x):
        pass

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass