import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
import sys
import model as modl


def read_feature(fname):
  return np.array(Image.open(fname), dtype=np.uint8)


classes = ['Author', 'Equation', 'FigureCaption', 'FigureText', 'Heading', 'PageFooter', 'PageHeader', 'PageNumber', 'Paragraph', 'References', 'Sparse', 'Subtitle', 'TableCaption', 'TableFooter', 'TableSparseColumnHeader', 'TableSparseMulticolumn', 'Title']
print(classes)
train_features = []
train_targets = []
for label,cl in enumerate(classes):
    features = glob("/home/amit/experiments/tabio/train/"+cl+"/*.png")
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
train_dataloader = data.DataLoader(train_dataset,batch_size=10,shuffle=True)

device = torch.device("cuda")
model = modl.model
model = model.to(device)
criterion = nn.NLLLoss()# Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)
epochs = 30
for e in range(epochs):
    running_loss = 0
    loss_len = 0
    for images, labels in train_dataloader:
        images,labels = images.to(device),labels.to(device)
        if modl.model_type=="MLP":
            images = images.view(images.shape[0], -1)
        else:
            images = images[:,None,:,:]

    
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

device= torch.device("cpu")
model=model.to(device)
torch.save(model, './trained_net.pth')

