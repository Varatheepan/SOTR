from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import copy
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys

################################################
	# Importing functions from other directory
################################################
sys.path.append('../../utils')
from core50_dataset_class import *
from core50_utils import *

#################################################
	# Loading aruguments
#################################################
parser = argparse.ArgumentParser(description='CORe50 dataset args')

parser.add_argument('--label_id', type = int, default = 2,
					help = 'label_id = 2 for object level and label_id = 3 for category level')

parser.add_argument('--shifter', type = int, default = 1,
					help = 'shifter = 1 for object level and shifter = 5 for category levelan argument in calculating rough index ')

parser.add_argument('--number_of_classes', type = int, default = 50,
					help = 'number_of_classes = 50 for object level and number_of_classes = 10 for category level')

parser.add_argument('--task_label_id', type = int, default = 5,
					help = 'an argument in calculating rough index')

args = parser.parse_args()

images_dir = '/content/core50_128x128/'
train_csv_file = '/content/core50_128x128/train_image_path_and_labels-1.csv'
test_csv_file = '/content/core50_128x128/test_image_path_and_labels-1.csv'

################################################
	# Training parameters
################################################
batch_size = 50
step_size = 20
gamma = 1
number_of_tasks = 5
num_classes_in_task = 10
number_of_classes = 50
learning_rate = 0.01
num_epochs = 60
mode = 'incremental_nc'

################################################
	#  Transformations and data loading
################################################
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([transforms.Resize(256),
								transforms.CenterCrop(224),
								transforms.ToTensor(),
								transforms.Normalize(mean = mean, std = std),])
CORe50_dataset = CORe50(train_csv_file, test_csv_file, images_dir, args)
test = CORe50_dataset.get_test_set()
test = test[::20]
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)
test_loader_size = len(test)

################################################
	# Training and Testing
################################################
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, number_of_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()

fc_ = ['fc.weight','fc.bias']
fc_parameters =  list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in fc_, model.named_parameters()))))

optimizer = optim.SGD(fc_parameters, lr = learning_rate, momentum = 0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

for task_id in range(0, number_of_tasks):
	print('###   Train on task', str(task_id), '  ###')
	train = []
	for class_id in range(task_id * num_classes_in_task, (task_id + 1) * num_classes_in_task):
		train += CORe50_dataset.get_train_by_task(mode, class_id)
	train = train[::20]	
	# train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
	train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
	train_model(model, criterion, optimizer, scheduler, train_loader,  num_epochs, transform = transform)
number_of_correct_by_class, number_by_class, confusion_matrix = test_model(model, criterion, optimizer, test_loader, test_loader_size, number_of_classes, transform = transform)

torch.save(model.state_dict(),'naive_model/naive2.pt')
