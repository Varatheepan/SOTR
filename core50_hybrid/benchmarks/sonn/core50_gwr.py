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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
					help = 'shifter = 1 for object level and shifter = 5 for category level an argument in calculating rough index ')

parser.add_argument('--number_of_classes', type = int, default = 50,
					help = 'number_of_classes = 50 for object level and number_of_classes = 10 for category level')

parser.add_argument('--task_label_id', type = int, default = 5,
					help = 'an argument in calculating rough index')
args = parser.parse_args()

images_dir = '../../../datasets/core50_128x128/'
train_csv_file = '../../../datasets/core50_128x128/train_image_path_and_labels.csv'
test_csv_file = '../../../datasets/core50_128x128/test_image_path_and_labels.csv'

################################################ss
	# Training parameters
################################################
number_of_classes = 50
number_of_tasks = 10
num_classes_in_task = 5
mode = 'incremental_nc'

batch_size = 100
step_size = 7
gamma = 0.5
cnn_gamma = 1
learning_rate = 0.005
cnn_learning_rate = 0.001
num_epochs = 1
weight_dict = {}
bias_dict = {}

act_thr = np.exp(-9)
fir_thr = 0.03
eps_b = 0.09	# moving BM nodes according to activation
eps_n = 0.005	# moving neighbour nodes according to activation
tau_b = 3.7		# firing
tau_n = 17.3
alpha_b = 1.01
alpha_n = 1.01
h_0 = 1
sti_s = 1
lab_thr = 0.5
max_age = 1000
max_size = 8000
random_state = None
gwr_epochs = 1

gwr_imgs_skip = 1
k = 10				#KNN k value

Include_FC = True
individual_class = False
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
test = test[::5] #reducing the frame rate from 20 fps to 5 fps
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False)
test_loader_size = len(test)

################################################
	# Training and Testing
###############################################

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes_in_task)
model = model.to(device)
# feature_model = nn.Sequential(*list(model.children())[:-1])		# removes last fc layer keeping average pooloing layer
fc_ = ['fc.weight','fc.bias']
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
cnn_parameters = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in fc_, model.named_parameters()))))
fc_parameters =  list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in fc_, model.named_parameters()))))

optimizer = optim.SGD(fc_parameters, lr = learning_rate, momentum = 0.9)
cnn_optimizer = optim.SGD(cnn_parameters, lr = cnn_learning_rate, momentum = 0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
cnn_scheduler = lr_scheduler.StepLR(cnn_optimizer, step_size = step_size, gamma = cnn_gamma)

gwr_model = gwr_torch(act_thr, fir_thr, eps_b, eps_n, tau_b, tau_n, alpha_b, alpha_n,
							h_0, sti_s,lab_thr, max_age, max_size, random_state = None)

start_time_o = time.time()
for task_id in range(0, number_of_tasks):
	print('###   Train on task', str(task_id), '  ###')
	train = []
	for class_id in range(task_id * num_classes_in_task, (task_id + 1) * num_classes_in_task):
		train += CORe50_dataset.get_train_by_task(mode, class_id)
	train = train[::20]	
	# input('wait :-)')
	train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
	start_time = time.time()
	if Include_FC:
		model, gwr_model = train_gwr_fc(model, gwr_model, criterion, optimizer, cnn_optimizer, scheduler, cnn_scheduler, train_loader,
						num_classes_in_task, task_id, gwr_epochs,num_epochs, gwr_imgs_skip, transform = transform)
		end_time = time.time()
		print("Time taken for the task : ",end_time - start_time)
		weight_dict[task_id] = copy.deepcopy(model.fc.weight.data[...])
		bias_dict[task_id] = copy.deepcopy(model.fc.bias.data[...])
	else:
		model, gwr_model = train_gwr(model, gwr_model,train_loader,num_classes_in_task, task_id, gwr_epochs, num_epochs,
			gwr_imgs_skip, individual_class = individual_class, transform = transform)
		end_time = time.time()
		print("Time taken for the task : ",end_time - start_time)
end_time_o = time.time()

if Include_FC:
	gwr_overall,gwr_categorical_acc,mh_overall,mh_cl_bycl,nodes_per_task = test_gwr_fc(model, gwr_model, criterion, test_loader, test_loader_size, number_of_tasks,
					num_classes_in_task, weight_dict, bias_dict,k, transform = transform)
	# end_time_o = time.time()
else:
	if not individual_class:
		gwr_overall,gwr_categorical_acc,nodes_per_task = test_gwr(model, gwr_model, test_loader, test_loader_size,
			number_of_tasks,num_classes_in_task, k, individual_class = individual_class,transform = transform)
	else:
		gwr_overall,gwr_categorical_acc,nodes_per_task = test_gwr(model, gwr_model, test_loader, test_loader_size,
			50,num_classes_in_task, k, individual_class = individual_class,transform = transform)
print('GWR Parameters: ')
print('act_thr: ', act_thr,' fir_thr: ', fir_thr, ' eps_b: ', eps_b, ' eps_n: ', eps_n,' tau_b: ', tau_b, ' tau_n: ', tau_n, ' gwr_epochs: ', gwr_epochs)
if Include_FC:
	print('FC Parameters: ')
	print('Lr: ',learning_rate, 'gamma: ',gamma,' batch_size: ',batch_size,'step_size: ',step_size,' epochs: ',num_epochs)

print("Time taken fro the whole training : ",end_time_o - start_time_o)
print('GWR nodes per task: ', nodes_per_task)
print("GWR categorical accuracy: ", gwr_categorical_acc )
print("GWR overall accuracy: ", gwr_overall)
if Include_FC:
	print("MH class by class accuracy: ", mh_cl_bycl)
	print("MH overall accuracy: ", mh_overall)

np.savez('trained_gwr_model_individual_1.pt', gwr_model)

# print('gwr_acc: ',gwr_acc)
# print('gwr_categorical_acc: ',gwr_categorical_acc)

# TAU_B = [3,5,7]
# TAU_N = [12.5,15,17.5]

ACT_THR = [10]
FIR_THR = [0.05,0.03,0.01]

# EPS_B = [0.05,0.03,0.01]
# EPS_N = [0.01,0.001,0.005]

'''time_list = []
acc_list_act_thr = []
# acc_list_eps_b =[]
for act_thr in ACT_THR:	
	print('act_thr: ',act_thr)
	acc_list_fir_thr = []
	for fir_thr in FIR_THR:
		# input('wait :-)')
		print('fir_thr: ',fir_thr)
		model = models.resnet18(pretrained=True)
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, num_classes_in_task)
		model = model.to(device)
		# feature_model = nn.Sequential(*list(model.children())[:-1])		# removes last fc layer keeping average pooloing layer
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
		scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)


		gwr_model = gwr_torch(np.exp(-act_thr), fir_thr, eps_b, eps_n, tau_b, tau_n, alpha_b, alpha_n,
									h_0, sti_s,lab_thr, max_age, max_size, random_state = None)
		start_time_o = time.time()
		for task_id in range(0, number_of_tasks):
			print('###   Train on task', str(task_id), '  ###')
			train = []
			for class_id in range(task_id * num_classes_in_task, (task_id + 1) * num_classes_in_task):
				train += CORe50_dataset.get_train_by_task(mode, class_id)
			# train = train[::10]	/
			# input('wait :-)')
			train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
			start_time = time.time()
			if Include_FC:
				train_gwr_fc(model, gwr_model, criterion, optimizer, cnn_optimizer, scheduler, cnn_scheduler, train_loader,
								num_classes_in_task, task_id, gwr_epochs,num_epochs, gwr_imgs_skip, transform = transform)
				end_time = time.time()
				print("Time taken for the task : ",end_time - start_time)
				weight_dict[task_id] = copy.deepcopy(model.fc.weight.data[...])
				bias_dict[task_id] = copy.deepcopy(model.fc.bias.data[...])
			else:
				train_gwr(model, gwr_model,optimizer,train_loader,num_classes_in_task, task_id, gwr_epochs, num_epochs, gwr_imgs_skip, 
							transform = transform)
				end_time = time.time()
				print("Time taken for the task : ",end_time - start_time)

		end_time_o = time.time()
		if Include_FC:
			gwr_overall,gwr_categorical_acc,mh_overall,mh_cl_bycl,nodes_per_task = test_gwr_fc(model, gwr_model, criterion, test_loader, test_loader_size, number_of_tasks,
							num_classes_in_task, weight_dict, bias_dict,k, transform = transform)
		else:
			gwr_overall,gwr_categorical_acc,nodes_per_task = test_gwr(model, gwr_model, test_loader, test_loader_size, 
												number_of_tasks,num_classes_in_task, k,transform = transform)

		print('GWR Parameters: ')
		print('act_thr: ', act_thr,' fir_thr: ', fir_thr, ' eps_b: ', eps_b, ' eps_n: ', eps_n,' tau_b: ', tau_b, ' tau_n: ', tau_n, ' gwr_epochs: ', gwr_epochs)
		if Include_FC:
			print('FC Parameters: ')
			print('Lr: ',learning_rate, 'gamma: ',gamma,' batch_size: ',batch_size,'step_size: ',step_size,' epochs: ',num_epochs)

		print("Time taken fro the whole training : ",end_time_o - start_time_o)
		print('GWR nodes per task: ', nodes_per_task)
		print("GWR categorical accuracy: ", gwr_categorical_acc )
		print("GWR overall accuracy: ", gwr_overall)
		if Include_FC:
			print("MH class by class accuracy: ", mh_cl_bycl)
			print("MH overall accuracy: ", mh_overall)

		time_list.append(end_time_o - start_time_o)
		acc_list_fir_thr.append(gwr_overall)
	acc_list_act_thr.append(acc_list_fir_thr)
print('Feature vector size: ',num_ftrs)

print('ACT_THR:', ACT_THR)
print('FIR_THR: ',FIR_THR)
print(acc_list_act_thr)'''

# print('EPS_B:', EPS_B)
# print('EPS_N: ',EPS_N)
# print(acc_list_eps_b)
