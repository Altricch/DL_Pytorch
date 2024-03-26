# CODE ADAPTED FROM REGULAR FEED FORWARD NN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #Relu, Tanh etc.
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ------------ Create Network ------------- #
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(CNN, self).__init__()
        # Same convolution with padding and stride (nothing changes)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride = (1,1), padding = (1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride = (1,1), padding = (1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        
        return x
    
# model = NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x).shape)
        
# ------------ Set Device ------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------ Hyperparameters ------------- #
IN_CHANNELS = 1
CLASSES= 10
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 20


# ------------ Load Data ------------- #
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train= False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ------------ INITIALIZE NETWORK ------------- #
model = CNN().to(device)

# ------------ LOSS FUNCTION ------------- #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ------------ Train Loop ------------- #
for epoch in range(EPOCHS):
    print("EPOCH", epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backward
        # Set all gradients to 0 for each batch, s.t. it doesnt store the back prop calculations
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent 
        optimizer.step()
        
        
# ------------ Check Accuracy ------------- #

def check_accuracy(loader, model):
    num_correct = 0
    num_sampels = 0
    
    model.eval()
    
    # No gradent computation needed
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_sampels += predictions.size(0)
        
        print(f"Got {num_correct} / {num_sampels} with accuracy {float(num_correct)/float(num_sampels)*100}") 
    
    model.train()
        
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)