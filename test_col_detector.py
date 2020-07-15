import torch
import numpy as np
from torch.utils import data
from torch import nn
from PIL import Image
from glob import glob
import model as modl
import pickle
from sklearn.metrics import confusion_matrix
import data_loader
import frontend
import column_detection as cd
import detect_columns
import config

all_col_labels = []
all_col_preds = []
for page in list(data_loader.test_pages())[:100]:
	print(page)
	labeled_boxes = frontend.read_labels(page)
	lines = frontend.read_lines(page)
	# def npmap(x): np.array(x, dtype=np.uint8)
	masks = frontend.stage1(lines)
	col_preds = detect_columns.detect_columns(masks)
	
	col_labs = [cd.fake_column_detection(l, labeled_boxes) for l in lines]
	all_col_labels.extend(col_labs)
	all_col_preds.extend(col_preds)

all_col_labels = [config.col_classes[x] for x in all_col_labels]

cf = confusion_matrix(all_col_labels,all_col_preds)
print(cf)
# print('Accuracy: {100.0 * correct / total}%')