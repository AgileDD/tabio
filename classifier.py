import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F

# generate 1000 samples of size 10X100
Samples = 1000
M = 10
N = 100

train_features = np.random.rand(Samples,M,N)
train_targets = np.random.randint(0,1,Samples)

torch_train_X = torch.from_numpy(train_features)
torch_train_Y = torch.from_numpy(train_targets)

train_dataset = data.TensorDataset(torch_train_X,torch_train_Y)
train_dataloader = data.DataLoader(train_dataset)

model = nn.Sequential(nn.Linear(1000, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))# Define the loss
model = model.double()
criterion = nn.NLLLoss()# Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in train_dataloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("Training loss: "+str(running_loss/len(train_dataloader)))


