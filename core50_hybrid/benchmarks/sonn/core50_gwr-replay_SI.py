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
import pickle
import random

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

images_dir = '/content/core50_128x128/'
train_csv_file = '/content/core50_128x128/train_image_path_and_labels-1.csv'
test_csv_file = '/content/core50_128x128/test_image_path_and_labels-1.csv'

################################################
	#Modified model
################################################

class ResnetExtended(nn.Module):
	def __init__(self,pretrained_model,num_ftrs,num_classes_in_task):
		super(ResnetExtended, self).__init__()
		self.pretrained_model = pretrained_model
		self.pretrained_model.fc = nn.Linear(num_ftrs,1024)
		self.fc2 = nn.Linear(1024,1024)
		self.fc3 = nn.Linear(1024,num_classes_in_task)

	def forward(self, x):
		x = self.pretrained_model(x)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

################################################ss
	# Training parameters
################################################

number_of_classes = 50
number_of_tasks = 10
num_classes_in_task = 5
mode = 'incremental_nc'

#FC and CNN parameters
batch_size = 100
step_size = 7
gamma = 0.95
cnn_gamma = 1
learning_rate = 0.01
cnn_learning_rate = 0.001
num_epochs = 10
weight_dict = {}
bias_dict = {}

#SI parameters and and matrices
previous_parameters = {}
previous_task_data = {}
small_omega = {}
epsilon = 1e-3
scaling_factor = 1e0

#replay
replay_samples_per_task = 50

#GWR parameters
act_thr = np.exp(-10)
fir_thr = 0.03
eps_b = 0.09	# moving BM nodes according to activation
eps_n = 0.005	# moving neighbour nodes according to activation
tau_b = 3.7		# firing
tau_n = 17.3
alpha_b = 1.01
alpha_n = 1.01
h_0 = 1
sti_s = 1
max_age = 1000
max_size = 8000
random_state = None
gwr_epochs = 5

gwr_imgs_skip = 1
k = 10				#KNN k value

Include_FC = True
individual_class = False
Incoporate_SI = True
gwr_trained = True
replay_gwr = False
replay_fc = False
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
# test = test[::20] #reducing the frame rate 
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False)
test_loader_size = len(test)

################################################
	# Training and Testing
###############################################

original_model = models.resnet18(pretrained=True)
num_ftrs = original_model.fc.in_features

model_copy = copy.deepcopy(original_model)
feature_model = nn.Sequential(*list(model_copy.children())[:-1])		# removes last fc layer keeping average pooloing layer
feature_model = feature_model.to(device)
model = ResnetExtended(original_model,num_ftrs,num_classes_in_task)
model = model.to(device)
# print(model)

last_fc = ['fc3.weight','fc3.bias','pretrained_model.fc.weight','pretrained_model.fc.bias','fc2.weight','fc2.bias']
tuned_fc = ['pretrained_model.fc.weight','pretrained_model.fc.bias','fc2.weight','fc2.bias']
criterion = nn.CrossEntropyLoss()

parameters =  list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in last_fc, model.named_parameters()))))
tuned_parameters = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in tuned_fc, model.named_parameters()))))

#optimizers and schedulers
# optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
# scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
optimizer = optim.SGD(parameters, lr = learning_rate, momentum = 0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
tuned_optimizer = optim.SGD(tuned_parameters, lr = cnn_learning_rate, momentum = 0.9)
tuned_scheduler = lr_scheduler.StepLR(tuned_optimizer, step_size = step_size, gamma = cnn_gamma)

if gwr_trained:
  with open('trained_gwr_model_2.pkl','rb') as input:
    gwr_model = pickle.load(input)
else:
  gwr_model = gwr_torch(act_thr, fir_thr, eps_b, eps_n, tau_b, tau_n, alpha_b, alpha_n,
                h_0, sti_s, max_age, max_size, random_state = None)

#samples for replay
replay_samples = []

start_time_o = time.time()
for task_id in range(0, number_of_tasks):
  task_replay_samples = []
  print('###   Train on task', str(task_id), '  ###')
  train = []
  for class_id in range(task_id * num_classes_in_task, (task_id + 1) * num_classes_in_task):
    task_data = CORe50_dataset.get_train_by_task(mode, class_id)
    train += task_data
    rand_indexes = random.sample(range(0, len(task_data)), replay_samples_per_task)
    task_replay_samples.extend([task_data[c] for c in rand_indexes])
  replay_samples.extend(task_replay_samples)  
  # train = train[::10]	
  train_loader1 = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
  # if replay_fc:
    # train.extend(replay_samples) #adding replay samples to train fc
  train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
  start_time = time.time()
  if Include_FC:	#train Multihead
    if Incoporate_SI:	#train model with si on higher cnn layers
      if task_id == 0:
        for name, parameter in model.named_parameters():
          name = name.replace('.', '_')
          previous_parameters[name] = parameter.data.clone()
      model, gwr_model = train_gwr_si(model, feature_model, gwr_model, criterion, optimizer, tuned_optimizer, scheduler, tuned_scheduler,
              train_loader,train_loader1, previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor,
              tuned_parameters, num_classes_in_task, task_id, gwr_epochs,num_epochs, gwr_imgs_skip, gwr_trained, transform = transform)
      big_omega_update(model, previous_task_data, small_omega, epsilon, task_id, tuned_parameters)
    else:	#train gwr and Fc layers only
      model, gwr_model = train_gwr_fc(model, gwr_model, criterion, optimizer, tuned_optimizer, scheduler, tuned_scheduler,
            train_loader, num_classes_in_task, task_id, gwr_epochs,num_epochs, gwr_imgs_skip, gwr_trained, replay_fc, replay_samples, 20, transform = transform)

    end_time = time.time()
    print("Time taken for the task : ",end_time - start_time)
    weight_dict[task_id] = copy.deepcopy(model.fc3.weight.data[...])
    bias_dict[task_id] = copy.deepcopy(model.fc3.bias.data[...])
  else:		#train gwr only
    if not gwr_trained:
      model, gwr_model = train_gwr(model, gwr_model,train_loader,num_classes_in_task, task_id, gwr_epochs, num_epochs,
        gwr_imgs_skip, gwr_trained, individual_class = individual_class, transform = transform)
      end_time = time.time()
      print("Time taken for the task : ",end_time - start_time)
end_time_o = time.time()

if replay_gwr:
  print('GWR being replayed ...')
  replay_loader = torch.utils.data.DataLoader(replay_samples, batch_size = batch_size, shuffle = False)
  model, gwr_model = train_gwr(model, gwr_model,replay_loader,num_classes_in_task, task_id, gwr_epochs, num_epochs,
    gwr_imgs_skip, gwr_trained, individual_class = individual_class, transform = transform)

if Include_FC:
  gwr_overall,gwr_categorical_acc,mh_overall,mh_cl_bycl,nodes_per_task = test_gwr_fc(model, feature_model, gwr_model, criterion, test_loader, test_loader_size, number_of_tasks,
              num_classes_in_task, weight_dict, bias_dict,k, transform = transform)
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

# np.savez('trained_gwr_model__full_dataset_1.pt', gwr_model)

'''

# TAU_B = [3,5,7]
# TAU_N = [12.5,15,17.5]

# ACT_THR = [10]
# FIR_THR = [0.05,0.03,0.01]

EPSILON = [0.001,0.01,0.1]	#10**i for i in range(-3,0)]
SCALING_FACTOR = [10**i for i in range(-3,4)] #50,150,200,300,500,750]

# EPS_B = [0.05,0.03,0.01]
# EPS_N = [0.01,0.001,0.005]

time_list = []
acc_list_epsilon = []
# acc_list_scaling_factor =[]
for epsilon in EPSILON:	
	acc_list_scaling_factor = []
	for scaling_factor in SCALING_FACTOR:
		print('epsilon: ',epsilon, ' scaling factor: ',scaling_factor)
		original_model = models.resnet18(pretrained=True)
		num_ftrs = original_model.fc.in_features

		model_copy = copy.deepcopy(original_model)
		feature_model = nn.Sequential(*list(model_copy.children())[:-1])		# removes last fc layer keeping average pooloing layer
		feature_model = feature_model.to(device)
		model = ResnetExtended(original_model,num_ftrs,num_classes_in_task)
		model = model.to(device)
		# print(model)

		last_fc = ['fc3.weight','fc3.bias','pretrained_model.fc.weight','pretrained_model.fc.bias','fc2.weight','fc2.bias']
		tuned_fc = ['pretrained_model.fc.weight','pretrained_model.fc.bias','fc2.weight','fc2.bias']
		criterion = nn.CrossEntropyLoss()

		parameters =  list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in last_fc, model.named_parameters()))))
		tuned_parameters = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in tuned_fc, model.named_parameters()))))

		#optimizers and schedulers
		# optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
		# scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
		optimizer = optim.SGD(parameters, lr = learning_rate, momentum = 0.9)
		scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
		tuned_optimizer = optim.SGD(tuned_parameters, lr = cnn_learning_rate, momentum = 0.9)
		tuned_scheduler = lr_scheduler.StepLR(tuned_optimizer, step_size = step_size, gamma = cnn_gamma)

		if gwr_trained:
			with open('trained_gwr_model_2.pkl','rb') as input:
				gwr_model = pickle.load(input)
		else:
			gwr_model = gwr_torch(act_thr, fir_thr, eps_b, eps_n, tau_b, tau_n, alpha_b, alpha_n,
										h_0, sti_s, max_age, max_size, random_state = None)

		#samples for replay
		replay_samples = []

		start_time_o = time.time()
		for task_id in range(0, number_of_tasks):
			task_replay_samples = []
			print('###   Train on task', str(task_id), '  ###')
			train = []
			for class_id in range(task_id * num_classes_in_task, (task_id + 1) * num_classes_in_task):
				task_data = CORe50_dataset.get_train_by_task(mode, class_id)
				train += task_data
				rand_indexes = random.sample(range(0, len(task_data)), replay_samples_per_task)
				task_replay_samples.extend([task_data[c] for c in rand_indexes])
				
			train = train[::20]	
			train_loader1 = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
			if replay_fc:
				train.extend(replay_samples) #adding replay samples to train fc
			train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
			start_time = time.time()
			if Include_FC:	#train Multihead
				if Incoporate_SI:	#train model with si on higher cnn layers
					if task_id == 0:
						for name, parameter in model.named_parameters():
							name = name.replace('.', '_')
							previous_parameters[name] = parameter.data.clone()
					model, gwr_model = train_gwr_si(model, feature_model, gwr_model, criterion, optimizer, tuned_optimizer, scheduler, tuned_scheduler,
									train_loader,train_loader1, previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor,
									tuned_parameters, num_classes_in_task, task_id, gwr_epochs,num_epochs, gwr_imgs_skip, gwr_trained, transform = transform)
					big_omega_update(model, previous_task_data, small_omega, epsilon, task_id, tuned_parameters)
				else:	#train gwr and Fc layers only
					model, gwr_model = train_gwr_fc(model, gwr_model, criterion, optimizer, tuned_optimizer, scheduler, tuned_scheduler,
								train_loader, num_classes_in_task, task_id, gwr_epochs,num_epochs, gwr_imgs_skip, transform = transform)

				end_time = time.time()
				print("Time taken for the task : ",end_time - start_time)
				weight_dict[task_id] = copy.deepcopy(model.fc3.weight.data[...])
				bias_dict[task_id] = copy.deepcopy(model.fc3.bias.data[...])
			else:		#train gwr only
				if not gwr_trained:
					model, gwr_model = train_gwr(model, gwr_model,train_loader,num_classes_in_task, task_id, gwr_epochs, num_epochs,
						gwr_imgs_skip, gwr_trained, individual_class = individual_class, transform = transform)
					end_time = time.time()
					print("Time taken for the task : ",end_time - start_time)
		end_time_o = time.time()

		if replay_gwr:
			print('GWR being replayed ...')
			replay_loader = torch.utils.data.DataLoader(replay_samples, batch_size = batch_size, shuffle = False)
			model, gwr_model = train_gwr(model, gwr_model,replay_loader,num_classes_in_task, task_id, gwr_epochs, num_epochs,
				gwr_imgs_skip, gwr_trained, individual_class = individual_class, transform = transform)

		if Include_FC:
			gwr_overall,gwr_categorical_acc,mh_overall,mh_cl_bycl,nodes_per_task = test_gwr_fc(model, feature_model, gwr_model, criterion, test_loader, test_loader_size, number_of_tasks,
									num_classes_in_task, weight_dict, bias_dict,k, transform = transform)
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

		# np.savez('trained_gwr_model_individual_1.pt', gwr_model)
		time_list.append(end_time_o - start_time_o)
		acc_list_scaling_factor.append(gwr_overall)
	acc_list_epsilon.append(acc_list_scaling_factor)

# print('ACT_THR:', ACT_THR)
# print('FIR_THR: ',FIR_THR)
print('EPSILON:', EPSILON)
print('SCALING_FACTOR: ',SCALING_FACTOR)
print(acc_list_epsilon)
'''