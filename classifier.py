import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
import sys

M = 20
N = 100

def read_feature(fname):
  return np.array(Image.open(fname), dtype=np.uint8)


classes = glob('/home/brian/line_data/train/*')
classes = [c.split("/")[-1] for c in classes]
print(classes)
train_features = []
train_targets = []
for label,cl in enumerate(classes):
    features = glob("/home/brian/line_data/train/"+cl+"/*.png")
    targets = [label]*len(features)
    train_features = train_features + list(map(read_feature, features))
    train_targets = train_targets + targets
train_features = np.array(train_features)/128.0
train_targets = np.array(train_targets)

print(train_targets.shape)
print(train_features.shape)


Samples = len(train_targets)

torch_train_X = torch.from_numpy(train_features)
torch_train_Y = torch.from_numpy(train_targets)

train_dataset = data.TensorDataset(torch_train_X,torch_train_Y)
train_dataloader = data.DataLoader(train_dataset,batch_size=200,shuffle=True)

def weights_init_normal(m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
            m.bias.data.fill_(0)

def init_weights(m):
        if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform(m.weight)
                    m.bias.data.fill_(0.01)

model = nn.Sequential(nn.Linear(M*N, 128),
                      nn.BatchNorm1d(128),
                      nn.Dropout(0.2),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.BatchNorm1d(64),
                      nn.Dropout(0.2),
                      nn.ReLU(),
                      nn.Linear(64, 17),
                      nn.LogSoftmax(dim=1))# Define the loss
model = model.double()
model.apply(weights_init_normal)
criterion = nn.NLLLoss()# Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)
epochs = 200
for e in range(epochs):
    running_loss = 0
    loss_len = 0
    for images, labels in train_dataloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        ### print output
        ### print labels
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_len += 1
    print("Training loss: "+str(running_loss/loss_len))
    # print("Training loss: "+str(loss.item()))

torch.save(model, './trained_net.pth')

