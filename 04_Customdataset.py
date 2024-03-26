import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


# DATASET:
# https://www.kaggle.com/datasets/c75fbba288ac0418f7786b16e713d2364a1a27936e63f4ec47502d73d6ef30ab



class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        image = io.imread(image_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        if self.transform:
            image = self.transform(image)
            
        return (image, y_label)
    
    
################################################
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F #Relu, Tanh etc.
    from torch.utils.data import DataLoader
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torchvision

    # model = NN(784, 10)
    # x = torch.randn(64, 784)
    # print(model(x).shape)
            
    # ------------ Set Device ------------- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE", device)


    # ------------ Hyperparameters ------------- #
    IN_CHANNELS = 3
    CLASSES= 10
    LR = 0.001
    BATCH_SIZE = 64
    EPOCHS = 20


    # ------------ Load Data ------------- #
    dataset = CatsAndDogsDataset("cats_dogs.csv", root_dir="cats_dogs_resized", transform = transforms.ToTensor())
    LENGTH = len(dataset)
    splittrain = int(LENGTH * 0.8)
    splittest = int(LENGTH * 0.2)
    print("TRAIN", splittrain, "TEST", splittest)
    trainset, testset = torch.utils.data.random_split(dataset, [splittest, splittrain])
    train_loader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True)


    # ------------ INITIALIZE NETWORK ------------- #
    model = torchvision.models.googlenet(pretrained = True)
    model.to(device)

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
            
        check_accuracy(train_loader,model)
        check_accuracy(test_loader,model)
            
