import csv_file
import config
import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from PIL import Image
import os.path
import sys
import data_loader
import frontend


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
    model = torch.load(os.path.join(os.path.dirname(__file__), './col_trained_net.pth'))
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


def train():
    classes = config.col_classes
    all_masks = []
    all_col_labels = []
    for page in list(data_loader.training_pages())[:1]:
        print(page)
        labeled_boxes = frontend.read_labels(page)
        lines = frontend.read_lines(page)
        # def npmap(x): np.array(x, dtype=np.uint8)
        masks = map(lambda m: m.resize((200, 20), resample=Image.BICUBIC), frontend.stage1(lines))
        masks = list(map(np.array, masks))
        col_labs = [fake_column_detection(l, labeled_boxes) for l in lines]
        all_masks.extend(list(masks))
        all_col_labels.extend(col_labs)

    labels = [classes[x] for x in all_col_labels]
    train_features = np.array(all_masks)/128.0
    train_targets = np.array(labels)

    print(train_targets.shape)
    print(train_features.shape)


    Samples = len(train_targets)

    torch_train_X = torch.from_numpy(train_features)
    torch_train_Y = torch.from_numpy(train_targets)

    train_dataset = data.TensorDataset(torch_train_X,torch_train_Y)
    train_dataloader = data.DataLoader(train_dataset,batch_size=10,shuffle=True)

    device = torch.device("cuda")
    model = ColumnModel()
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
    model.__module__ = 'column_detection'
    torch.save(model, './col_trained_net.pth')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        train()
