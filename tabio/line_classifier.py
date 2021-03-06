import os.path
import pickle
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils import data

import tabio.column_detection
import tabio.config
import tabio.data_loader
import tabio.frontend
import tabio.lexical
import tabio.metrics

M = 20
N = 100


def vcat_with_check(a, b):
    if len(a) == 0:
        return b
    else:
        return np.vstack((a, b))


class LineModel(nn.Module):
    def __init__(self):
        super(LineModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3,stride=2)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d()
        # Good for scaling 128x128 images
        self.dense1 = nn.Linear(in_features=1408, out_features=512)

        self.dense1_bn = nn.BatchNorm1d(512)
        self.dense2 = nn.Linear(512+200, len(tabio.config.mapped_classes))
        # self.dense2 = nn.Linear(512, len(config.mapped_classes))
        self.double()

    def forward(self, x, textf):
        if textf.size()[1]!=200:
            print(f"Warning: Found only {textf.size()[1]} dimensions in lexical model, padding ...")
        pad = nn.ConstantPad1d((0,200-textf.size()[1]),0.0)
        textf = pad(textf)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv5_bn(self.conv5(x)), 2))
        x = x.view(-1, self.num_flat_features(x))  # reshape
        # print(x.shape)
        x = F.relu(self.dense1_bn(self.dense1(x)))
        # x = F.relu(self.dense2(x))
        x = F.relu(self.dense2(torch.cat((x, textf), dim=1)))
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def load(path):
    model = torch.load(os.path.join(path, 'trained_net.pt'))
    model.eval()
    return model


# given a list of features each representing 1 line
# this evaluates teh score of each class for each feature
def eval(model, features, lexical_features):
    print("Running eval")
    features = list(map(np.array, features))
    features = np.array(features)/128.0
    targets = np.array([0]*len(features))

    torch_test_X = torch.from_numpy(features)
    torch_test_T = torch.from_numpy(lexical_features)
    torch_test_Y = torch.from_numpy(targets)

    test_dataset = data.TensorDataset(torch_test_X, torch_test_T, torch_test_Y)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=10, shuffle=False)

    all_scores = []

    for feature_set, text_feat, labels in test_dataloader:
        feature_set = feature_set[:, None, :, :]
        outputs = model(feature_set, text_feat)
        # print(outputs)
        # print(outputs.shape[0])
        tuner = torch.tensor([tabio.config.tune]*outputs.shape[0])
        outputs = outputs + tuner
        print(f"{outputs=}")
        all_scores.extend(list(outputs))
    return all_scores


def prepare_data(pages, lexical_path):
    lexical_model = tabio.lexical.load(lexical_path)
    classes = tabio.config.classes
    train_features = []
    train_targets = []
    max_pages = 10000
    i_page = 0
    train_textf = []
    for page in pages:
        if i_page % 100 == 0:
            print(i_page)
        i_page = i_page+1
        if i_page > max_pages:
            continue
        labeled_boxes = tabio.frontend.read_labels(page)
        def column_detector(lines, masks): return [
            tabio.column_detection.fake_column_detection(l, labeled_boxes) for l in lines]
        features, lines = tabio.frontend.create(page, column_detector)
        line_classifications = [tabio.column_detection.read_line_classification(
            line, labeled_boxes) for line in lines]

        usable_features = []
        usable_label_indexes = []
        usable_lines = []
        for f, c, li in zip(features, line_classifications, lines):
            # filter out lines that have no manual label
            if c is not None:
                # get rid of the column classification, and only keep the line class
                line_class = tabio.config.interpret_label(c)[1]
                if line_class not in tabio.config.class_map.keys():
                    print("Warning: "+line_class+" not in class dicitonary")
                    continue
                try:
                    # class_index = classes.index(line_class)
                    class_index = tabio.config.mapped_classes.index(
                        tabio.config.class_map[line_class])
                    usable_features.append(f)
                    usable_label_indexes.append(class_index)
                    usable_lines.append(li)
                except ValueError:
                    continue
        if len(usable_features) == 0 or len(usable_lines) == 0:
            continue

        text_features = tabio.lexical.create_lexical_features(
            lexical_model, usable_lines)
        print(text_features.shape)
        train_textf = vcat_with_check(train_textf, text_features)
        print(train_textf.shape)
        usable_features = list(map(np.array, usable_features))
        train_features.extend(usable_features)
        train_targets.extend(usable_label_indexes)

    train_features = np.array(train_features)/128.0
    print(train_features.shape)
    print(train_textf.shape)
    train_targets = np.array(train_targets)

    Samples = len(train_targets)

    torch_train_X = torch.from_numpy(train_features)
    torch_train_T = torch.from_numpy(train_textf)
    torch_train_Y = torch.from_numpy(train_targets)

    train_dataset = data.TensorDataset(
        torch_train_X, torch_train_T, torch_train_Y)
    return data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)


def train(model_path):
    print("line classifier training")

    print("preparing data...")
    train_dataloader = prepare_data(tabio.data_loader.training_pages(), model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LineModel()
    model = model.to(device)
    # Optimizers require the parameters to optimize and a learning rate
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)
    epochs = 30
    for e in range(epochs):
        running_loss = 0
        loss_len = 0
        for images, text_feat, labels in train_dataloader:
            images, textf, labels = images.to(
                device), text_feat.to(device), labels.to(device)
            images = images[:, None, :, :]
            # Training pass
            optimizer.zero_grad()
            output = model(images, textf)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_len += 1
        print("Training loss: "+str(running_loss/loss_len))

    device = torch.device("cpu")
    model = model.to(device)
    model.__module__ = 'line_classifier'
    torch.save(model, os.path.join(model_path, 'trained_net.pt'))


def test():
    classes = tabio.config.classes
    test_dataloader = prepare_data(tabio.data_loader.test_pages())
    model = load()
    correct = 0
    total = 0
    allref = []
    allhyp = []
    tuner = torch.tensor([tabio.config.tune]*10)
    all_proba = []
    for images, textf, labels in test_dataloader:
        images = images[:, None, :, :]
        outputs = model(images, textf)
        print(outputs)
        outputs = outputs + tuner
        _, predicted = torch.max(outputs, 1)
        all_proba = vcat_with_check(all_proba, outputs.detach().numpy())
        for i in predicted:
            print(i)
        print('')
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        allref.extend(list(labels))
        allhyp.extend(list(predicted))
        print(labels)
        print(predicted)
    fpr, tpr = tabio.metrics.roc_curve(allref, all_proba, [0])
    pickle.dump([fpr, tpr], open("roc.pt", "wb"))
    tp = 0
    fn = 0
    fp = 0
    mapped_classes = tabio.config.mapped_classes
    for i in range(len(allref)):
        if "Tab" in mapped_classes[allref[i]] or 'Frame' in mapped_classes[allref[i]]:
            if "Tab" in mapped_classes[allhyp[i]] or 'Frame' in mapped_classes[allhyp[i]]:
                tp = tp + 1.0
            else:
                fn = fn + 1.0
        else:
            if "Tab" in mapped_classes[allhyp[i]] or 'Frame' in mapped_classes[allhyp[i]]:
                fp = fp + 1.0
    print("Precision="+str(tp/(tp+fp)))
    print("Recall="+str(tp/(tp+fn)))
    cf = confusion_matrix(allref, allhyp)
    print(cf)
    pickle.dump(cf, open("cf.pt", "wb"))
    print(correct)
    print(total)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        train(sys.argv[2])
    else:
        test()
