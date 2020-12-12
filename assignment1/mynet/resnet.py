import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

def flatten(x):
    N = x.shape[0]
    return x.view(N,-1)
def train(model, optimizer, epochs=1, loader_train, loader_val, print_every = 100):
    model = model.to(device=device)
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype = dtype)
            y = y.to(device=device, dtype = torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f'%(t,loss.item()))
                check(loader_val,model)
                print()
def check(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct+= (preds==y).sum()
            num_samples+= preds.size(0)
        acc = float(num_correct)/num_samples
        print('Got %d / %d correct (%.2f)'%(num_correct, num_samples, 100*acc))
class block(nn.Module):
    def __init__(self, in_planes, planes):
        super(block, self).__init__()
        #conv-bn-shortcut
        self.conv1 = nn.Conv2d(in_planes,planes,3, padding=1)
        self.conv2 = nn.Conv2d(planes,planes,3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if planes != in_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,planes,3, padding=1),
                nn.BatchNorm2d(planes)
                ) 
    def forward(self,x):
        out = F.relu((self.bn1(self.conv1(x))))
        out = F.relu((self.bn2(self.conv2(out)))+ self.shortcut(x))
        return out
class resNet(nn.Module):
    def __init__(self,in_channels,hidden_channels,num_classes, width = 32, height = 32):
        super(resNet,self).__init__()
        self.conv = nn.Conv2d(in_channels,hidden_channels[0], 3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels[0])
        self.res1 = block(hidden_channels[0],hidden_channels[1])
        self.res2 = block(hidden_channels[1],hidden_channels[2])
        self.res3 = block(hidden_channels[2],hidden_channels[3])
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(hidden_channels[3]*int(width/2)*int(height/2), num_classes)
    def forward(self,x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.maxpool(out)
        out = self.fc(flatten(out))
        return out
class dblock(nn.Module):
    def __init__(self,in_planes,planes):
        super(dblock, self).__init__()
        