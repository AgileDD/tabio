import torch
import numpy as np
from torch.utils import data
from torch import nn
from PIL import Image
from glob import glob
import model as modl
import pickle
from sklearn.metrics import confusion_matrix
import config
import line_classifier

def read_feature(fname):
  return np.array(Image.open(fname), dtype=np.uint8)

classes = config.classes
print(classes)
test_features = []
test_targets = []
for label,cl in enumerate(classes):
    features = glob("/home/amit/experiments/tabio/test/"+cl+"/*.png")
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
test_dataloader = data.DataLoader(test_dataset,batch_size=10)


model = line_classifier.load()

correct = 0
total = 0
allref = []
allhyp = []
for images, labels in test_dataloader:
    if modl.model_type=="MLP":
        images = images.view(images.shape[0], -1)
    else:
        images = images[:,None,:,:]
    outputs = model(images)
    print(outputs)

    _, predicted = torch.max(outputs, 1)
    for i in predicted:
        print(i)
    print('')
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    allref.extend(list(labels))
    allhyp.extend(list(predicted))
    print(labels)
    print(predicted)

tp=0
fn=0
fp=0
for i in range(len(allref)):
    if "Tab" in classes[allref[i]]:
        print(classes[allref[i]])
        if "Tab" in classes[allhyp[i]]:
            tp = tp + 1.0
        else:
            fn = fn + 1.0
    else:
        if "Tab" in classes[allhyp[i]]:
            fp = fp + 1.0
print("Precision="+str(tp/(tp+fp)))
print("Recall="+str(tp/(tp+fn)))
cf = confusion_matrix(allref,allhyp)
print(cf)


pickle.dump(cf,open("cf.pickle","wb"))


print(correct)
print(total)
# print('Accuracy: {100.0 * correct / total}%')
