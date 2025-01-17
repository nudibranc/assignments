import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914), (0.2023))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.MNIST(
    './cs231n/datasets',
    train=True, download=True,
    transform=transform
)
loader_train = DataLoader(
    cifar10_train,
    batch_size=64,
    sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN))
)
cifar10_val = dset.MNIST(
    './cs231n/datasets',
    train=True, download=True,
    transform=transform
)
loader_val = DataLoader(
    cifar10_val, batch_size=64, 
    sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000))
)

cifar10_test = dset.CIFAR10(
    './cs231n/datasets',
    train=False, download=True,
    transform=transform
)
loader_test = DataLoader(cifar10_test, batch_size=64)

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def train_part34(model, optimizer, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()
def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
class Block(nn.Module):
    def __init__(self, in_planes, planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding =1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding =1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if (in_planes != planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,planes,3,padding = 1),
                nn.BatchNorm2d(planes)
            )
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))
        return out
class ResNet(nn.Module):
    def __init__(self,in_channels, hidden_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels[0], 3, padding = 1)
        self.bn = nn.BatchNorm2d(hidden_channels[0])
        self.res1 = Block(hidden_channels[0], hidden_channels[1])
        self.res2 = Block(hidden_channels[1], hidden_channels[2])
        self.res3 = Block(hidden_channels[2], hidden_channels[3])
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(hidden_channels[3]*14*14, num_classes)
    def forward(self,x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.maxpool(out)
        out = self.fc(flatten(out))
        return out
