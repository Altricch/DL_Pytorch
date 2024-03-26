# CODE ADAPTED FROM REGULAR FEED FORWARD NN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #Relu, Tanh etc.
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

# ------------ Set Device ------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE", device)

# ------------ Hyperparameters ------------- #
IN_CHANNELS = 3
CLASSES = 10
LR = 0.001
BATCH_SIZE = 1024
EPOCHS = 5


# ------------ Load Pretrained Model ------------- #
import sys

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
model = torchvision.models.vgg16(weights="DEFAULT")
print(model)

# Freeze layers for no back prop
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100,10))
# print(model)
model = model.to(device)



# ------------ Load Data ------------- #
train_dataset = datasets.CIFAR10(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ------------ INITIALIZE NETWORK ------------- #
# model = CNN().to(device)

# ------------ LOSS FUNCTION ------------- #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


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
        

# check_accuracy(test_loader,model)

# ------------ Train Loop ------------- #
for epoch in range(EPOCHS):
    print("EPOCH", epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):
        # print("BIDX",batch_idx)
        data = data.to(device)
        # print("data", type(data))
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
    
    check_accuracy(train_loader,model)
        
