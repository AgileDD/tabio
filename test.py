import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from PIL import Image
from glob import glob

def read_feature(fname):
  return np.array(Image.open(fname), dtype=np.uint8)

test_paragraph_features = glob('test/Paragraph/*.png')
test_sparce_features = []#glob('test/TableSparseMulticolumn/*.png')

test_features = np.array(list(map(read_feature, test_paragraph_features))+list(map(read_feature, test_sparce_features)))/128.0
test_targets = np.array([1]*len(test_paragraph_features) + [0]*len(test_sparce_features))

torch_test_X = torch.from_numpy(test_features)
torch_test_Y = torch.from_numpy(test_targets)

test_dataset = data.TensorDataset(torch_test_X,torch_test_Y)
test_dataloader = data.DataLoader(test_dataset)


model = torch.load('./trained_net.pth')

correct = 0
total = 0
for images, labels in test_dataloader:
    images = images.view(images.shape[0], -1)
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    for i in predicted:
        print(i)
    print('')
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(correct)
print(total)
print(f'Accuracy: {100.0 * correct / total}%')