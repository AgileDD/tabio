import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
import sys
model_type="CNN"

M = 20
N = 200
if model_type=="MLP":
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
                          nn.Linear(64, 3),
                          nn.LogSoftmax(dim=1))# Define the loss
    model = model.double()
    model.apply(weights_init_normal)

elif model_type=="CNN":
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
            self.conv1_bn = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.conv2_bn = nn.BatchNorm2d(64)
    
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
            self.conv5_bn = nn.BatchNorm2d(64)
            self.dropout = nn.Dropout2d()
            self.dense1 = nn.Linear(in_features=36864, out_features=512)# Good for scaling 128x128 images
            
            self.dense1_bn = nn.BatchNorm1d(512)
            self.dense2 = nn.Linear(512, 3)
    
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
    model=Net()
    model = model.double()

