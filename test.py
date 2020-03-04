import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
import model as modl

def read_feature(fname):
  return np.array(Image.open(fname), dtype=np.uint8)

classes = glob('/home/brian/line_data/test/*')
classes = [c.split("/")[-1] for c in classes]
print(classes)
test_features = []
test_targets = []
for label,cl in enumerate(classes):
    features = glob("/home/brian/line_data/test/"+cl+"/*.png")
    targets = [label]*len(features)
    test_features = test_features + list(map(read_feature, features))
    test_targets = test_targets + targets
test_features = np.array(test_features)/128.0
test_targets = np.array(test_targets)


### test_paragraph_features = []#glob('/home/brian/line_data/test/Paragraph/*.png')
### test_sparce_features =  glob('/home/brian/line_data/test/TableSparseMulticolumn/*.png')
### 
### test_features = np.array(list(map(read_feature, test_paragraph_features))+list(map(read_feature, test_sparce_features)))/128.0
### test_targets = np.array([1]*len(test_paragraph_features) + [0]*len(test_sparce_features))

torch_test_X = torch.from_numpy(test_features)
torch_test_Y = torch.from_numpy(test_targets)

test_dataset = data.TensorDataset(torch_test_X,torch_test_Y)
test_dataloader = data.DataLoader(test_dataset,batch_size=200)


model = torch.load('./trained_net.pth')

model.eval()

correct = 0
total = 0
for images, labels in test_dataloader:
    if modl.model_type=="MLP":
        images = images.view(images.shape[0], -1)
    else:
        images = images[:,None,:,:]
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    for i in predicted:
        print(i)
    print('')
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print(labels)
    print(predicted)

print(correct)
print(total)
# print('Accuracy: {100.0 * correct / total}%')