import torch.nn as nn
from torch.nn import functional as F
from GaborNet import GaborConv2d

class GaborNN(nn.Module):
    def __init__(self):
        super(GaborNN, self).__init__()
        self.g0 = GaborConv2d(in_channels=3, out_channels=32, kernel_size=(15, 15), stride=1)
        self.c1 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3, 3),stride=1)
        self.c2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3, 3),stride=1)
        self.c3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3, 3),stride=1)
        self.c4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3, 3),stride=1)
        self.fc1=nn.Linear(in_features=3200,out_features=128)
        self.fc2=nn.Linear(in_features=128,out_features=128)
        self.fc3=nn.Linear(in_features=128,out_features=2)
    def forward(self, x):
        x = F.relu(self.g0(x))
        x = nn.MaxPool2d(kernel_size=(2, 2))(x)
        x = F.relu(self.c1(x))
        x = nn.MaxPool2d(kernel_size=(2, 2))(x)
        x = F.relu(self.c2(x))
        x = nn.MaxPool2d(kernel_size=(2, 2))(x)
        x = F.relu(self.c3(x))
        x = nn.MaxPool2d(kernel_size=(2, 2))(x)
        x = F.relu(self.c4(x))
        x = nn.MaxPool2d(kernel_size=(2, 2))(x)
        x=x.view(-1,128*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x