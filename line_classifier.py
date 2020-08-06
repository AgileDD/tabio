import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
import sys
import config
import os.path

M = 20
N = 100

class LineModel(nn.Module):
    def __init__(self):
        super(LineModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d()
        self.dense1 = nn.Linear(in_features=17664, out_features=512)# Good for scaling 128x128 images
        
        self.dense1_bn = nn.BatchNorm1d(512)
        self.dense2 = nn.Linear(512, 17)
        self.double()

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = x.view(-1, self.num_flat_features(x)) #reshape
        x = F.relu(self.dense1_bn(self.dense1(x)))
        x = F.relu(self.dense2(x))
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def load():
    model = torch.load(os.path.join(os.path.dirname(__file__), './trained_net.pth'))
    model.eval()
    return model


def read_feature(fname):
  return np.array(Image.open(fname), dtype=np.uint8)

def train():
    classes = config.classes
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
    train_dataloader = data.DataLoader(train_dataset,batch_size=10,shuffle=True, drop_last=True)

    device = torch.device("cuda")
    model = LineModel()
    model = model.to(device)
    criterion = nn.NLLLoss()# Optimizers require the parameters to optimize and a learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)
    epochs = 30
    for e in range(epochs):
        running_loss = 0
        loss_len = 0
        for images, labels in train_dataloader:
            images,labels = images.to(device),labels.to(device)
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


if __name__ == '__main__':
    train()