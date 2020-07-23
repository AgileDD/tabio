import csv_file
import config
import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from PIL import Image


def read_line_classification(line, labeled_boxes):
    for bbox in line.bboxes:
        for l in labeled_boxes:
            if csv_file.is_bbox_inside(l.bbox, bbox):
                return l.name
    return None

def fake_column_detection(line, labeled_boxes):
    classification = read_line_classification(line, labeled_boxes)
    if classification is None:
        return None
    return classification.split('-')[0]


class ColumnModel(nn.Module):
    def __init__(self):
        super(ColumnModel, self).__init__()
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
    model = torch.load('./col_trained_net.pth')
    model.eval()
    return model

# given a list of masks (one mask per line on a page) classify each one as single or double column
# returns a list of classifications
def eval(model, masks):
    ms = map(lambda m: m.resize((200, 20), resample=Image.BICUBIC), masks)
    ms = list(map(np.array, ms))
    labels = [0]*len(ms)
    test_features = np.array(ms)/128.0
    test_targets = np.array(labels)

    #print(test_targets.shape)
    #print(test_features.shape)


    Samples = len(test_targets)

    torch_test_X = torch.from_numpy(test_features)
    torch_test_Y = torch.from_numpy(test_targets)

    test_dataset = data.TensorDataset(torch_test_X,torch_test_Y)
    test_dataloader = data.DataLoader(test_dataset,batch_size=10,shuffle=False)

    allhyp = []

    for images, labels in test_dataloader:
        images = images[:,None,:,:]
        outputs = model(images)
        #print(outputs)

        _, predicted = torch.max(outputs, 1)
        #for i in predicted:
        #    print(i)
        #print('')
        allhyp.extend(list(predicted))
        #print(predicted)=
    
    return map(lambda h: config.col_class_inference[int(h.numpy())], allhyp)