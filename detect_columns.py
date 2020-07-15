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

def detect_columns(masks):
	ms = map(lambda m: m.resize((200, 20), resample=Image.BICUBIC), masks)
	ms = list(map(np.array, ms))
	labels = [0]*len(ms)
	test_features = np.array(ms)/128.0
	test_targets = np.array(labels)

	print(test_targets.shape)
	print(test_features.shape)


	Samples = len(test_targets)

	torch_test_X = torch.from_numpy(test_features)
	torch_test_Y = torch.from_numpy(test_targets)

	test_dataset = data.TensorDataset(torch_test_X,torch_test_Y)
	test_dataloader = data.DataLoader(test_dataset,batch_size=10,shuffle=False)


	model = torch.load('./col_trained_net.pth')

	model.eval()

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
		allhyp.extend(list(predicted))
		print(predicted)

	return allhyp

