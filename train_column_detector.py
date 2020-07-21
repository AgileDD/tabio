import frontend
import config
import data_loader
import column_detection as cd
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import sys
from torch import nn
import torch.nn.functional as F



classes = config.col_classes
all_masks = []
all_col_labels = []
for page in list(data_loader.training_pages())[:1000]:
    print(page)
    labeled_boxes = frontend.read_labels(page)
    lines = frontend.read_lines(page)
    # def npmap(x): np.array(x, dtype=np.uint8)
    masks = map(lambda m: m.resize((200, 20), resample=Image.BICUBIC), frontend.stage1(lines))
    masks = list(map(np.array, masks))
    col_labs = [cd.fake_column_detection(l, labeled_boxes) for l in lines]
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
model = cd.ColumnModel()
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
torch.save(model, './col_trained_net.pth')
