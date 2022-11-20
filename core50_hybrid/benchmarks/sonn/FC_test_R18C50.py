from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import pickle
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.append('../../utils')
from core50_dataset_class import *
# from core50_utils import *

#################################################
	# Loading aruguments
#################################################
parser = argparse.ArgumentParser(description='CORe50 dataset args')

parser.add_argument('--label_id', type = int, default = 2,
					help = 'label_id = 2 for object level and label_id = 3 for category level')

parser.add_argument('--shifter', type = int, default = 1,
					help = 'shifter = 1 for object level and shifter = 5 for category level an argument in calculating rough index ')

parser.add_argument('--number_of_classes', type = int, default = 50,
					help = 'number_of_classes = 50 for object level and number_of_classes = 10 for category level')

parser.add_argument('--task_label_id', type = int, default = 5,
					help = 'an argument in calculating rough index')
args = parser.parse_args()


images_dir = '/content/core50_128x128/'
train_csv_file = '/content/core50_128x128/train_image_path_and_labels-1.csv'
test_csv_file = '/content/core50_128x128/test_image_path_and_labels-1.csv'

################################################
	#Modified model
################################################

class ResnetExtended(nn.Module):
	def __init__(self,pretrained_model,num_ftrs,num_classes_in_task):
		super(ResnetExtended, self).__init__()
		self.pretrained = pretrained_model
		self.pretrained.fc = nn.Linear(num_ftrs,1024)
		self.fc2 = nn.Linear(1024,512)
		self.fc3 = nn.Linear(512,num_classes_in_task)

	def forward(self, x):
		x = F.relu(self.pretrained(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

################################################
		#Functions
################################################
def read_load_inputs_labels(image_names, labels, transform = None):
	img_inputs = []
	for i in range(len(image_names)):
		image = io.imread(image_names[i])
		if transform:
			image = transform(Image.fromarray(image))
		img_inputs.append(image)
	inputs = torch.stack(img_inputs)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	inputs = inputs.to(device)
	labels = labels.to(device)
	return inputs, labels
def train_model(model, criterion, optimizer, scheduler, train_loader,  num_epochs, num_classes_in_task, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			labels = labels % num_classes_in_task
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
		scheduler.step()
	return model


def big_omega_update(model, previous_task_data, small_omega, epsilon, task_id,si_parameters):
	epsilon = torch.tensor(epsilon).cuda()
	print(epsilon)
	for name, parameter in model.named_parameters():
		if name in si_parameters:
			# print('correct bru')
			if parameter.requires_grad:
				name = name.replace('.', '_')
				current_task_parameter = parameter.detach().clone().cuda()
				if task_id == 0:
					big_omega = parameter.detach().clone().zero_().cuda()
					big_omega = big_omega + small_omega[name] / torch.add((torch.zeros(current_task_parameter.shape).cuda())**2, epsilon).cuda()
				else:
					prev_task_parameter = previous_task_data['{}_prev_task'.format(name)].cuda()
					big_omega = previous_task_data['{}_big_omega'.format(name)]
					big_omega += small_omega[name].cuda() / torch.add((current_task_parameter - prev_task_parameter)**2, epsilon).cuda()
				previous_task_data.update({'{}_prev_task'.format(name): current_task_parameter})
				previous_task_data.update({'{}_big_omega'.format(name): big_omega})

def surrogate_loss(model, previous_task_data,cnn_parameters):
	surr_loss = 0
	for name, parameter in model.named_parameters():
		if name in cnn_parameters:
			if parameter.requires_grad:	
				name = name.replace('.', '_')
				prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
				big_omega = previous_task_data['{}_big_omega'.format(name)]
				surr_loss += (torch.mul(big_omega, (parameter - prev_task_parameter)**2)).sum()
	# print('surr_loss: ', surr_loss)
	return surr_loss

def train_si(model, criterion, optimizer, scheduler, train_loader,
				previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor, tuned_parameter,
				num_classes_in_task, task_id, num_epochs, transform = None):

	model.train()
	for name, parameter in model.named_parameters():
		if name in tuned_parameter:
		  name = name.replace('.', '_')
		  small_omega[name] = parameter.data.clone().zero_()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			fc_labels = labels // num_classes_in_task
			# print(fc_labels)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)

				if task_id == 0:
					loss = criterion(outputs, fc_labels)  
					loss.backward()
				else:
					#loss = criterion(outputs, fc_labels) 
					loss = criterion(outputs, fc_labels) + scaling_factor * surrogate_loss(model, previous_task_data,tuned_parameter)
					#si_loss = criterion(outputs, fc_labels) + scaling_factor * surrogate_loss(model, previous_task_data,tuned_parameter)
					# si_loss.backward()
					# tuned_optimizer.step()
					loss.backward()

				for name, parameter in model.named_parameters():
					if name in tuned_parameter:
					  if parameter.requires_grad:
						  name = name.replace('.', '_')
						  small_omega[name].add_(-parameter.grad * (parameter.detach() - previous_parameters[name]))
						  previous_parameters[name] = parameter.detach().clone()
				optimizer.step()
		scheduler.step()
	return model





def test_model(model, criterion, optimizer, test_loader, test_loader_size, number_of_classes, transform = None):
	best_acc = 0.0
	model.eval()
	test_loss = 0.0
	corrects = 0
	number_of_classes = 2
	number_of_correct_by_class = np.zeros((1, number_of_classes), dtype = int)
	number_by_class = np.zeros((1, number_of_classes), dtype = int)
	confusion_matrix = np.zeros((number_of_classes, number_of_classes), dtype = int)
	for image_names, labels in test_loader:
		with torch.no_grad():
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			labels = labels // 5
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			# print(preds)
			loss = criterion(outputs, labels)
			test_loss += loss.item() * inputs.size(0)
			corrects += torch.sum(preds == labels.data)
			for k in range(preds.size(0)):
				confusion_matrix[labels[k]][preds[k]] = confusion_matrix[labels[k]][preds[k]] + 1
				number_by_class[0][labels[k]] = number_by_class[0][labels[k]] + 1
				if preds[k] == labels[k]:
					number_of_correct_by_class[0][preds[k]] = number_of_correct_by_class[0][preds[k]] + 1
	
	print('number_of_correct_by_class: ',number_of_correct_by_class)
	print('number_by_class: ',number_by_class)
	print('class by class accuray: ',number_of_correct_by_class/number_by_class)
	epoch_loss = test_loss / test_loader_size
	# print(corrects.item())
	# print(test_loader_size)
	epoch_acc = corrects.double() / test_loader_size
	print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
	return	number_of_correct_by_class, number_by_class, confusion_matrix

################################################ss
	# Training parameters
################################################

mode = 'incremental_nc'
number_of_classes = 50
num_classes_in_task = 5
#FC and CNN parameters
batch_size = 40
step_size = 7
gamma = 0.9
cnn_gamma = 1
learning_rate = 0.0001
cnn_learning_rate = 0.001
num_epochs = 1
task_set = 0
trail_no = 1

#SI parameters and and matrices
previous_parameters = {}
previous_task_data = {}
small_omega = {}
epsilon = 1e-3
scaling_factor = 1500000

################################################
	#  Transformations and data loading
################################################
print(' ')
print('trail no: ', trail_no)
print('Parameters: ')
print('lr: ', learning_rate, ' num epochs: ', num_epochs, ' scaling factor: ', scaling_factor, ' epsilon: ', epsilon)

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([transforms.Resize(256),
								transforms.CenterCrop(224),
								transforms.ToTensor(),
								transforms.Normalize(mean = mean, std = std),])

original_model = models.resnet18(pretrained=True)
num_ftrs = original_model.fc.in_features

CORe50_dataset = CORe50(train_csv_file, test_csv_file, images_dir, args)
# model = copy.deepcopy(original_model)
model = ResnetExtended(original_model,num_ftrs,2)
model = model.to(device)
# model2 = model2.to(device)
# for name, parameter in model.named_parameters():
	# print(name)

criterion = nn.CrossEntropyLoss()
Fcs = ['pretrained.fc.weight','pretrained.fc.bias','fc2.weight','fc2.bias', 'fc3.weight','fc3.bias']
tuned_Fcs = ['pretrained.fc.weight','pretrained.fc.bias','fc2.weight','fc2.bias']

# tuned_parameters = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in tuned_Fcs, model.named_parameters()))))
# print(tuned_parameters)
parameters =  list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in Fcs, model.named_parameters()))))
optimizer = optim.SGD(parameters, lr = learning_rate, momentum = 0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
print(model.parameters())
'''
test = []
for class_id in range(task_set * num_classes_in_task, (task_set + 1) * num_classes_in_task):
		test += CORe50_dataset.get_test_by_task(mode, class_id)
test = test[::20]
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False)
test_loader_size = len(test)

train = []
for class_id in range(task_set * num_classes_in_task, (task_set + 1) * num_classes_in_task):
		train += CORe50_dataset.get_train_by_task(mode, class_id)
# train = train[::20]
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

model = train_model(model,criterion,optimizer,scheduler,train_loader, num_epochs, num_classes_in_task, transform = transform)

number_of_correct_by_class, number_by_class, confusion_matrix = test_model(model, criterion, 
	optimizer, test_loader, test_loader_size, num_classes_in_task, transform = transform)
	'''


for task_id in range(task_set, task_set+2):
	print('###   Train on task', str(task_id), '  ###')
	train = []
	for class_id in range(task_id * num_classes_in_task, (task_id + 1) * num_classes_in_task):
		train += CORe50_dataset.get_train_by_task(mode, class_id)
	# train = train[::10]	
	train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
	if task_id == 0:
		for name, parameter in model.named_parameters():
			if name in tuned_Fcs:
				name = name.replace('.', '_')
				previous_parameters[name] = parameter.data.clone()
	model = train_si(model, criterion, optimizer, scheduler, train_loader,
				previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor, tuned_Fcs,
				num_classes_in_task, task_id, num_epochs, transform = transform)
	big_omega_update(model, previous_task_data, small_omega, epsilon, task_id, tuned_Fcs)
	# print(previous_task_data)
test = []
for class_id in range(task_set * num_classes_in_task, (task_set + 2) * num_classes_in_task):
	test += CORe50_dataset.get_test_by_task(mode, class_id)
# test = test[::20]
test_loader = torch.utils.data.DataLoader(test, batch_size = 100, shuffle = False)
test_loader_size = len(test)

number_of_correct_by_class, number_by_class, confusion_matrix = test_model(model, criterion, 
optimizer, test_loader, test_loader_size, num_classes_in_task, transform = transform)


# from __future__ import print_function, division
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms, utils
# import matplotlib.pyplot as plt
# import time
# import copy
# import pandas as pd
# from skimage import io, transform
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import sys
# import pickle
# import random

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# sys.path.append('../../utils')
# from core50_dataset_class import *
# from core50_utils import *

# #################################################
# 	# Loading aruguments
# #################################################
# parser = argparse.ArgumentParser(description='CORe50 dataset args')

# parser.add_argument('--label_id', type = int, default = 2,
# 					help = 'label_id = 2 for object level and label_id = 3 for category level')

# parser.add_argument('--shifter', type = int, default = 1,
# 					help = 'shifter = 1 for object level and shifter = 5 for category level an argument in calculating rough index ')

# parser.add_argument('--number_of_classes', type = int, default = 50,
# 					help = 'number_of_classes = 50 for object level and number_of_classes = 10 for category level')

# parser.add_argument('--task_label_id', type = int, default = 5,
# 					help = 'an argument in calculating rough index')
# args = parser.parse_args()


# images_dir = '/content/core50_128x128/'
# train_csv_file = '/content/core50_128x128/train_image_path_and_labels-1.csv'
# test_csv_file = '/content/core50_128x128/test_image_path_and_labels-1.csv'

# ################################################
# 	#Modified model
# ################################################

# class ResnetExtended(nn.Module):
# 	def __init__(self,pretrained_model,num_ftrs,num_classes_in_task,h1,h2):
# 		super(ResnetExtended, self).__init__()
# 		self.pretrained_model = pretrained_model
# 		self.pretrained_model.fc = nn.Linear(num_ftrs,h1)
# 		self.fc2 = nn.Linear(h1,h2)
# 		self.fc3 = nn.Linear(h2,num_classes_in_task)

# 	def forward(self, x):
# 		x = self.pretrained_model(x)
# 		x = F.relu(self.fc2(x))
# 		x = self.fc3(x)
# 		return x

# ################################################
# 		#Functions
# ################################################

# def train_model(model, criterion, optimizer, scheduler, train_loader,  num_epochs, num_classes_in_task, transform = None):
# 	model.train()
# 	for epoch in range(1, num_epochs + 1):
# 		print('Epoch : ', epoch)
# 		for image_names, labels in train_loader:
# 			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
# 			labels = labels % num_classes_in_task
# 			optimizer.zero_grad()
# 			with torch.set_grad_enabled(True):
# 				outputs = model(inputs)
# 				loss = criterion(outputs, labels)
# 				loss.backward()
# 				optimizer.step()
# 		scheduler.step()
# 	return model

# def test_model(model, criterion, optimizer, test_loader, test_loader_size, number_of_classes, transform = None):
# 	best_acc = 0.0
# 	model.eval()
# 	test_loss = 0.0
# 	corrects = 0
# 	number_of_correct_by_class = np.zeros((1, number_of_classes), dtype = int)
# 	number_by_class = np.zeros((1, number_of_classes), dtype = int)
# 	confusion_matrix = np.zeros((number_of_classes, number_of_classes), dtype = int)
# 	for image_names, labels in test_loader:
# 		with torch.no_grad():
# 			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
# 			labels = labels % num_classes_in_task
# 			outputs = model(inputs)
# 			_, preds = torch.max(outputs, 1)
# 			loss = criterion(outputs, labels)
# 			test_loss += loss.item() * inputs.size(0)
# 			corrects += torch.sum(preds == labels.data)
# 			for k in range(preds.size(0)):
# 				confusion_matrix[labels[k]][preds[k]] = confusion_matrix[labels[k]][preds[k]] + 1
# 				number_by_class[0][labels[k]] = number_by_class[0][labels[k]] + 1
# 				if preds[k] == labels[k]:
# 					number_of_correct_by_class[0][preds[k]] = number_of_correct_by_class[0][preds[k]] + 1
	
# 	print('number_of_correct_by_class: ',number_of_correct_by_class)
# 	print('number_by_class: ',number_by_class)
# 	print('class by class accuray: ',number_of_correct_by_class/number_by_class)
# 	epoch_loss = test_loss / test_loader_size
# 	# print(corrects.item())
# 	# print(test_loader_size)
# 	epoch_acc = corrects.double() / test_loader_size
# 	print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
# 	return	number_of_correct_by_class, number_by_class, confusion_matrix

# ################################################ss
# 	# Training parameters
# ################################################

# mode = 'incremental_nc'
# number_of_classes = 50
# num_classes_in_task = 5
# #FC and CNN parameters
# batch_size = 40
# step_size = 7
# gamma = 0.9
# cnn_gamma = 1
# learning_rate = 0.005
# cnn_learning_rate = 0.001
# num_epochs = 15
# task_set = 2
# h1 = 4096
# h2 = 1024
# trail_no = 1
# ################################################
# 	#  Transformations and data loading
# ################################################
# print(' ')
# print('trail no: ', trail_no)
# print('Parameters: ')
# print('lr: ', learning_rate, ' num epochs: ', num_epochs, 'hidden neurons: ', h1, ',',h2)
# mean = [ 0.485, 0.456, 0.406 ]
# std = [ 0.229, 0.224, 0.225 ]

# transform = transforms.Compose([transforms.Resize(256),
# 								transforms.CenterCrop(224),
# 								transforms.ToTensor(),
# 								transforms.Normalize(mean = mean, std = std),])

# CORe50_dataset = CORe50(train_csv_file, test_csv_file, images_dir, args)
# test = []
# for class_id in range(task_set * num_classes_in_task, (task_set + 1) * num_classes_in_task):
# 		test += CORe50_dataset.get_test_by_task(mode, class_id)
# # test = test[::20]
# test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False)
# test_loader_size = len(test)

# original_model = models.resnet18(pretrained=True)
# num_ftrs = original_model.fc.in_features

# # model = copy.deepcopy(original_model)
# model = ResnetExtended(original_model,num_ftrs,num_classes_in_task,h1,h2)
# model = model.to(device)
# # model2 = model2.to(device)

# criterion = nn.CrossEntropyLoss()
# Fcs = ['pretrained_model.fc.weight','pretrained_model.fc.bias','fc2.weight','fc2.bias', 'fc3.weight','fc3.bias']
# # Fcs = ['fc.weight', 'fc.bias']

# parameters =  list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in Fcs, model.named_parameters()))))
# optimizer = optim.SGD(parameters, lr = learning_rate, momentum = 0.9)
# scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

# train = []
# for class_id in range(task_set * num_classes_in_task, (task_set + 1) * num_classes_in_task):
# 		train += CORe50_dataset.get_train_by_task(mode, class_id)
# # train = train[::20]
# train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

# model = train_model(model,criterion,optimizer,scheduler,train_loader, num_epochs, num_classes_in_task, transform = transform)

# number_of_correct_by_class, number_by_class, confusion_matrix = test_model(model, criterion, 
# 	optimizer, test_loader, test_loader_size, num_classes_in_task, transform = transform)