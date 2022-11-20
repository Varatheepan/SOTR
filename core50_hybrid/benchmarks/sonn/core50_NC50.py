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

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################
	# Importing functions from other directory
################################################
sys.path.append('../../utils')
from core50_dataset_class import *
from core50_utils_AP import *

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


# class Extend_model(nn.Module):
# 	def __init__(self,out_features):
# 		super(Extend_model, self).__init__()
# 		self.out_features = out_features

# 		self.multi_cnns = nn.ModuleDict({})
# 		self.max_pool = nn.MaxPool2d(5,5)
# 		self.multi_fcs = nn.ModuleDict({})


# 	def forward(self,x,cnn=None,fc=None):
# 		if cnn == -1:
# 			X = []
# 			for i,layer in self.multi_cnns.items():
# 				x_ = layer(x)
# 				x_ = self.max_pool(x_)
# 				x_ = torch.reshape(x_,(1,-1))
# 				X.append(x_)
# 			X = torch.stack(X,dim=1)
# 			# print('X: ',X.shape)
# 			return X 
# 		elif cnn:
# 			x = self.multi_cnns[str(int(cnn))](x)
# 			x = self.max_pool(x)
# 			x = x.view(-1, self.out_features)
# 			if fc:
# 				x = self.multi_fcs[str(int(fc))](x)	
# 		elif fc:
# 			x = self.multi_fcs[str(int(fc))](x)
# 		return x

class Extend_model(nn.Module):
	def __init__(self,out_features):
		super(Extend_model, self).__init__()
		self.out_features = out_features

		self.multi_cnns = nn.ModuleDict({})
		self.max_pool = nn.MaxPool2d(5)
		self.avgpool = nn.AvgPool2d(5)
		self.multi_fcs = nn.ModuleDict({})
		self.maxpool2 = nn.MaxPool2d(7)
		self.avgpool2 = nn.AvgPool2d(7)
		self.relu = nn.ReLU()
		self.lp_pool = nn.LPPool2d(3,7)
		

	def forward(self,x,cnn=None,fc=None,replay_data=None):
		if cnn == -1:
			x = self.maxpool2(x)   #self.lp_pool(x)
			x = torch.reshape(x,(1,-1))				
			return x 
		elif cnn and fc:
			# x = self.multi_cnns[str(int(cnn))](x)
			x = self.maxpool2(x) #max_pool(x)
			x = self.relu(x)
			x = x.view(-1, 512)  #(-1, self.out_features)
			if replay_data != None:
				x = torch.cat((x,replay_data))
			x = self.multi_fcs[str(int(fc))](x)
		elif cnn:
			x_2 = self.maxpool2(x)  #self.lp_pool(x)  #self.maxpool2(x)
			x = torch.reshape(x_2,(-1,512))
		elif fc:
			x = self.multi_fcs[str(int(fc))](x)
		return x


# class Extend_model(nn.Module):
# 	def __init__(self,out_features):
# 		super(Extend_model, self).__init__()
# 		self.out_features = out_features

# 		self.multi_cnns = nn.ModuleDict({})
# 		self.max_pool = nn.MaxPool2d(5)
# 		self.avgpool = nn.AvgPool2d(5)
# 		self.multi_fcs = nn.ModuleDict({})
# 		self.SEfc1 = nn.ModuleDict({})
# 		self.SEfc2 = nn.ModuleDict({})
# 		self.maxpool2 = nn.MaxPool2d(7)
# 		self.avgpool2 = nn.AvgPool2d(7)
# 		self.relu = nn.ReLU()
# 		self.lp_pool = nn.LPPool2d(3,5)
		

# 	def forward(self,x,cnn=None,fc=None):
# 		if cnn == -1:
# 			X = []
# 			for i,layer in self.multi_cnns.items():			
# 				x_1 = layer(x)
# 				x_2 = self.avgpool(x_1)
# 				x_2 = torch.reshape(x_2,(-1,self.out_features))
# 				x_2 = self.SEfc1[str(int(i))](x_2)
# 				x_2 = self.relu(x_2)
# 				x_2 = self.SEfc2[str(int(i))](x_2)
# 				x_2 = torch.nn.functional.softmax(x_2,dim=1)
# 				x_1 = torch.mul(x_1,x_2.view(x_1.shape[0], x_1.shape[1], 1, 1))
# 				x_1 = self.relu(x_1)
# 				x_1 = self.max_pool(x_1)
# 				x_3 = self.maxpool2(x)
# 				x_3 = torch.reshape(x_3,(1,-1))	
# 				x_1 = torch.reshape(x_1,(1,-1))
# 				x_1 = torch.cat((x_1,x_3), dim=1)
# 				X.append(x_1)
# 			X = torch.stack(X,dim=1)
# 			return X 
# 		elif cnn and fc:
# 			x = self.multi_cnns[str(int(cnn))](x)
# 			x_1 = self.avgpool(x)
# 			x_1 = torch.reshape(x_1,(-1,self.out_features))
# 			x_1 = self.SEfc1[str(int(cnn))](x_1)
# 			x_1 = self.relu(x_1)
# 			x_1 = self.SEfc2[str(int(cnn))](x_1)
# 			x_1 = torch.nn.functional.softmax(x_1,dim=1)
# 			x = torch.mul(x,x_1.view(x.shape[0], x.shape[1], 1, 1))
# 			x = self.relu(x)
# 			x = self.max_pool(x)
# 			x = x.view(-1, self.out_features)
# 			x = self.multi_fcs[str(int(fc))](x)
# 			x = torch.nn.functional.softmax(x,dim=1)
# 		elif cnn:
# 			x_3 = self.maxpool2(x)
# 			x_3 = torch.reshape(x_3,(-1,512))	
# 			x = self.multi_cnns[str(int(cnn))](x)
# 			x_1 = self.avgpool(x)
# 			x_1 = torch.reshape(x_1,(-1,self.out_features))
# 			x_1 = self.SEfc1[str(int(cnn))](x_1)
# 			x_1 = self.relu(x_1)
# 			x_1 = self.SEfc2[str(int(cnn))](x_1)
# 			x_1 = torch.nn.functional.softmax(x_1,dim=1)
# 			x = torch.mul(x,x_1.view(x.shape[0], x.shape[1], 1, 1))
# 			x = self.relu(x)
# 			x = self.max_pool(x)
# 			x = torch.reshape(x,(-1,self.out_features))
# 			x = torch.cat((x,x_3), dim=1)

# 		elif fc:
# 			x = self.multi_fcs[str(int(fc))](x)
# 		return x


def gwr_features(feature_model,model,task_id,train_loader):
	feature_model.eval()
	model.eval()
	gwr_feature_tensor = torch.tensor([])
	gwr_labels = torch.tensor([])
	# gwr_feature_tensor = gwr_feature_tensor

	for image_names, task_labels,fc_labels in train_loader:
		inputs, fc_labels = read_load_inputs_labels(image_names, fc_labels, transform = transform)
		gwr_label = fc_labels
		with torch.set_grad_enabled(False):
			feature_inputs = feature_model(inputs)
			gwr_features = model(feature_inputs,cnn=task_id+1)#,fc=task_id+1)
			# model = model.cpu()
			gwr_feature_tensor = torch.cat((gwr_feature_tensor,gwr_features.cpu()))
			gwr_labels  = torch.cat((gwr_labels,gwr_label.cpu().float()))
		# print('labels: ',gwr_labels.shape)
	return gwr_feature_tensor.cpu(), gwr_labels 

def train_MultiNet(feature_model, model, criterion, optimizer, scheduler, train_loader, task_id, fc_num_epochs, gwr_epochs,num_classes_in_task,num_tasks,replay_data=None):
	feature_model.eval()
	model.train()
	sample_size = 50
	img_idx = 0
	for image_names, task_labels,fc_labels in train_loader:
		head_id = int(task_labels[0])
		# print(head_id)
		break
	for epoch in range(1, fc_num_epochs + 1):
		print('CNN model epoch----- : ', epoch)
		for image_names, task_labels,fc_labels in train_loader:
			inputs, fc_labels = read_load_inputs_labels(image_names, fc_labels, transform = transform)
			# with torch.set_grad_enabled(False):
			optimizer.zero_grad()
			feature_inputs = feature_model(inputs)
			if replay_data:
				# print('replay')
				replay_samples = torch.stack(([replay_data[img_idx+j][0] for j in range(sample_size)])).to(device)
				# print('replay_samples: ', replay_samples.shape, 'feature inputs: ', feature_inputs.shape)
				# feature_inputs = torch.cat((feature_inputs,replay_samples))
				replay_labels = torch.tensor([replay_data[img_idx+j][1] for j in range(sample_size)], dtype=torch.int64).to(device)
				fc_labels = torch.cat((fc_labels,replay_labels))
				if (img_idx+sample_size)>len(replay_data):
					img_idx = 0
					replay_data = random.shuffle(replay_data)
				with torch.set_grad_enabled(True):
					outputs = model(feature_inputs,cnn=head_id+1,fc=head_id+1,replay_data=replay_samples)
					# print(outputs.shape)
					loss = criterion(outputs, fc_labels)
					loss.backward()	
					optimizer.step()
			else:
				# print(feature_inputs.shape)
				with torch.set_grad_enabled(True):
					outputs = model(feature_inputs,cnn=head_id+1,fc=head_id+1)
					# print(outputs.shape)
					loss = criterion(outputs, fc_labels)
					loss.backward()	
					optimizer.step()
					# tuned_optimizer.step()
		scheduler.step()			
	return model

def test_MultiNet(feature_model, model, clf, nca, test_loader, test_loader_size, number_of_tasks,num_classes_in_task,KNN_k,gwr=False,FC=False):
	class_div = 1
	model.eval()
	feature_model.eval()
	overall_corrects = 0
	gwr_corrects = 0
	smax = nn.Softmax(dim=0)
	if gwr and FC:
		predicted_class_by_class = torch.zeros(number_of_tasks*num_classes_in_task//class_div)
		true_class_by_class = torch.zeros(number_of_tasks*num_classes_in_task//class_div)
		gwr_category_by_category = torch.zeros(number_of_tasks)
		true_category_by_category = torch.zeros(number_of_tasks)
		nodes_per_tasks = gwr_model.nodes_per_task(number_of_tasks)
		for image_names, task_label,fc_label in test_loader:
			inputs, fc_label = read_load_inputs_labels(image_names, fc_label, transform = transform)
			# fc_label = (labels % num_classes_in_task)//class_div
			gwr_label = task_label
			with torch.set_grad_enabled(False):
				feature_inputs = feature_model(inputs)
				output = model(feature_inputs,cnn=-1)#,fc=gwr_label)
			output = output.cpu()
			# pred_tasks = gwr_model.choose_task(output, number_of_tasks,KNN_k)
			pred_task = int(clf.predict(nca.transform(torch.reshape(output,(1,-1)))))
			# print('----------------------------')
			# print('pred_task: ',pred_tasks)
			# pred_task = torch.from_numpy(pred_task).clone().detach().type(torch.int32).item()
			# print('pred_task: ',pred_task)
			# print('gwr_label: ',gwr_label)
			true_category_by_category[int(gwr_label)] += 1

			# pred_task_smax = smax(torch.exp(-torch.tensor(list(pred_tasks.values()),dtype=torch.float32)))
			# pred_task_smax = torch.pow(smax(torch.tensor(list(pred_tasks.values()),dtype=torch.float32)),4)
			# pred_task_smax = smax(torch.tensor(list(pred_tasks.values()),dtype=torch.float32))
			# print('pred_task_smax: ',pred_task_smax)
			# combined_matric = []
			# for p,pred_task in enumerate(pred_tasks.keys()):
			# with torch.set_grad_enabled(False):
				# output = model(feature_inputs,cnn=pred_task+1,fc=pred_task+1)
			# 		output = torch.nn.functional.softmax(output,dim=1)
			# 		# print('output: ', output)
			# 		# combined_matric.append(output[0]*pred_task_smax[p])
			# 		combined_matric.append(output[0]*pred_tasks[pred_task])

			# combined_matric = torch.stack(combined_matric)
			# sorted_matric = torch.argsort(combined_matric.view(1,-1)[0])
			# print(combined_matric)
			# print(sorted_matric)
			# time.sleep(1)
			# print((sorted_matric//num_classes_in_task)[-1])
			# print(list(pred_tasks.keys())[int(sorted_matric[-1]//num_classes_in_task)])
			# pred_task = int(list(pred_tasks.keys())[sorted_matric[-1]//2])
			# preds = int(sorted_matric[-1]%2)
			# print(sorted_matric[-1]%2)
			# print(sorted_matric%2)
			# time.sleep(20)

			if pred_task == int(gwr_label):
				gwr_corrects += 1
				gwr_category_by_category[gwr_label] += 1
			with torch.set_grad_enabled(False):
				output = model(feature_inputs,cnn=pred_task+1,fc=pred_task+1)
			_, preds = torch.max(output, 1)
			true_class_by_class[num_classes_in_task*task_label + fc_label.cpu()] += 1
			# preds[0] = preds[0]//class_div
			if (pred_task == int(gwr_label)) and ((preds) == int(fc_label.data[0])):
				overall_corrects += 1
				predicted_class_by_class[pred_task*num_classes_in_task + preds] += 1
		nodes_per_task1 = gwr_model.nodes_per_task(number_of_tasks)
		return gwr_corrects / test_loader_size * 100.0,gwr_category_by_category/true_category_by_category * 100,\
		overall_corrects / test_loader_size * 100.0,predicted_class_by_class/true_class_by_class * 100, nodes_per_task1
	
	elif FC:
		predicted_class_by_class = torch.zeros(number_of_tasks*num_classes_in_task//class_div)
		true_class_by_class = torch.zeros(number_of_tasks*num_classes_in_task//class_div)
		for image_names, task_label,fc_label in test_loader:
			inputs, fc_label = read_load_inputs_labels(image_names, fc_label, transform = transform)
			# fc_label = (labels % num_classes_in_task)//class_div
			# task_label = (labels // num_classes_in_task)
			with torch.set_grad_enabled(False):
				feature_inputs = feature_model(inputs)
				output = model(feature_inputs,cnn=task_label+1,fc=task_label+1)#,fc=gwr_label)
				# print('output: ',output.shape)
			_, preds = torch.max(output, 1)
			true_class_by_class[task_label*num_classes_in_task + fc_label.cpu()] += 1
			# print('preds: ',preds,' labels: ',labels)
			if (preds[0] == fc_label.data[0]):
				overall_corrects += 1
				predicted_class_by_class[task_label*num_classes_in_task//class_div + preds[0].cpu()] += 1
		return overall_corrects / test_loader_size * 100.0,predicted_class_by_class/true_class_by_class * 100
#  gwr_raw #
################################################ss
	# Training parameters
################################################

number_of_classes = 50
number_of_tasks = 10
num_classes_in_task = 5
mode = 'incremental_nc2'

#FC and CNN parameters
num_fc_classes = 5
out_features = 256
batch_size = 20
step_size = 10
gamma = 0.1
cnn_gamma = 1
learning_rate = 0.1
cnn_learning_rate = 0.001
fc_num_epochs = 40
weight_dict = {}
bias_dict = {}

#SI parameters and and matrices
previous_parameters = {}
previous_task_data = {}
small_omega = {}
epsilon = 1e-3
scaling_factor = 1000

#replay
replay_samples_per_class = 100
replay_samples_per_batch = 5

#GWR parameters
act_thr = np.exp(-8)#np.exp(-2)
fir_thr = 0.03
eps_b = 0.1	# moving BM nodes according to activation
eps_n = 0.0095	# moving neighbour nodes according to activation
tau_b = 5		# firing
tau_n = 15
alpha_b = 1.01
alpha_n = 1.01
h_0 = 1
sti_s = 1
max_age = 100
max_size_per_task = 5000
random_state = None
gwr_epochs = 5

gwr_imgs_skip = 1
KNN_k= 10				#KNN k value

individual_class = 0
Incoporate_SI = 1
gwr_trained = 1
FC_trained = 1
replay_gwr = 0
replay_fc = 0

TestFC = 1
TestGWR = 1

model_no = 1
suffix = 'NC50-6'
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
# test = CORe50_dataset.get_test_set()
# test = test[::100] #reducing the frame rate 
# test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False)
# test_loader_size = len(test)

test = []
for i in range(50):
	test += CORe50_dataset.get_test_by_task(mode, i)[::20]
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False)
test_loader_size = len(test)

################################################
	# Training and Testing
###############################################
print('KNN bayesian || n: 8|| weight: \'distance\' || p: 1.2 || algorithm: \'auto\' || dataset- ::10')  #polynominal(2)- gamma: \'auto\'
print('with pca analysis transform & only fc at the end  || n_components = 200')  #svm gamma=\'auto\',kernel=\'rbf\'  with Linear discriminant analysis 
# print('10 class fc')      #resnet_model)
# print('Testing with weighted inverse distance matric for GWR')

resnet_model = models.resnet18(pretrained=True)

resnet_model = nn.Sequential(*list(resnet_model.children())[:-2])		# removes last fc layer keeping average pooloing layer
# resnet_model.add_module('8',torch.nn.AvgPool2d(3,2))
# resnet_model.add_module('8',torch.nn.MaxPool2d(3,2))

resnet_model.to(device)

MultiNet = Extend_model(out_features)

if FC_trained:
	for task_id in range(0, 10):
		MultiNet.multi_cnns.update([[str(task_id+1), nn.Conv2d(512,out_features,3,1)]])
		MultiNet.multi_fcs.update([[str(task_id+1), nn.Linear(512,num_fc_classes)]])
		# MultiNet.SEfc1.update([[str(task_id+1), nn.Linear(out_features,int(out_features/2))]])
		# MultiNet.SEfc2.update([[str(task_id+1), nn.Linear(int(out_features/2),out_features)]])
	name = 'NC50_model/model_dataset_full_' + str(model_no) + '_' + suffix + '.pt'
	MultiNet.load_state_dict(torch.load(name))
	MultiNet = MultiNet.to(device)
	# print(MultiNet)

criterion = nn.CrossEntropyLoss()

#optimizers and schedulers
# optimizer = optim.SGD(MultiNet.parameters(), lr = learning_rate, momentum = 0.9)
# scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
# optimizer = optim.SGD(parameters, lr = learning_rate, momentum = 0.9)
# scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
# tuned_optimizer = optim.SGD(tuned_parameters, lr = cnn_learning_rate, momentum = 0.9)
# tuned_scheduler = lr_scheduler.StepLR(tuned_optimizer, step_size = step_size, gamma = cnn_gamma)

if gwr_trained:
	name = 'NC50_model/gwr_model_' + str(model_no) + '_' + suffix + '.pkl'
	with open(name,'rb') as input:
		gwr_model = pickle.load(input)
else:
	gwr_model = gwr_task_torch(act_thr, fir_thr, eps_b, eps_n, tau_b, tau_n, alpha_b, alpha_n,
								h_0, sti_s, max_age, max_size_per_task, random_state = None)

if not FC_trained:
	for i in range(1,11):
		MultiNet.multi_cnns.update([[str(i), nn.Conv2d(512,out_features,3,1)]])
		MultiNet.multi_fcs.update([[str(i), nn.Linear(512,num_fc_classes)]])

	MultiNet = MultiNet.to(device)

head_id_occ = [] #head id occurrance
start_time_o = time.time()
for task_id in range(0, 50):
	print('###   Train on task', str(task_id), '  ###')

	train = CORe50_dataset.get_train_by_task(mode, task_id)
	train = train[::20]	
	train_loader1 = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
	train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
	for image_names, task_labels,fc_labels in train_loader:
		head_id = int(task_labels[0])
		# print(head_id)
		print('object label: ', int(head_id*5 + fc_labels[0]+1))
		break
	if head_id in head_id_occ:
		replay_Data = []
		replay_data = gwr_model.getpos()[head_id]
		replay_labels = gwr_model.getlabels()[head_id]
		for i in range(len(replay_data)):
			# print(replay_data[i].shape)
			replay_Data.append([replay_data[i],replay_labels[i]])
		# print(len(replay_Data))
		# replay_Data = torch.FloatTensor(replay_Data)
		random.shuffle(replay_Data)

	start_time = time.time()

	optimizer = optim.SGD([\
			{'params':MultiNet.multi_fcs[str(head_id+1)].parameters()}],\
			# {'params':MultiNet.SEfc1[str(task_id+1)].parameters()},\
			# {'params':MultiNet.SEfc2[str(task_id+1)].parameters()}],\
			lr = learning_rate, momentum = 0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

	if not FC_trained:
		if head_id in head_id_occ:
			# print('with replay')
			MultiNet = train_MultiNet(resnet_model, MultiNet, criterion, optimizer, scheduler, train_loader, task_id, fc_num_epochs, gwr_epochs,num_classes_in_task,number_of_tasks,replay_Data)
		else:
			MultiNet = train_MultiNet(resnet_model, MultiNet, criterion, optimizer, scheduler, train_loader, task_id, fc_num_epochs, gwr_epochs,num_classes_in_task,number_of_tasks)



	if not gwr_trained:
		gwr_feature_tensor, gwr_labels = gwr_features(resnet_model,MultiNet,task_id,train_loader1)
		# print(gwr_feature_tensor.shape, gwr_labels.shape)
		if head_id in head_id_occ:
			gwr_model.train(gwr_feature_tensor, gwr_labels, head_id, n_epochs = gwr_epochs,new_task = False)	# GWR initiation
		else:
			gwr_model.train(gwr_feature_tensor, gwr_labels,  head_id, n_epochs = gwr_epochs, new_task = True)	# continuing

	if head_id not in head_id_occ:
		head_id_occ.append(head_id)
	print('head_id_occ: ',head_id_occ)

	torch.cuda.empty_cache()
	# print(gwr_model.getlabels()[head_id])#(torch.cuda.memory_allocated())

end_time_o = time.time()

if not FC_trained:
	torch.save(MultiNet.state_dict(),'NC50_model/model_dataset_full_'  + str(model_no) + '_' + suffix + '.pt')

if not gwr_trained:
	name = 'NC50_model/gwr_model_' + str(model_no)+ '_' + suffix + '.pkl'
	with open(name, 'wb') as output:
		pickle.dump(gwr_model, output, pickle.HIGHEST_PROTOCOL)

inputvectors = torch.tensor([])
svmlabels = torch.tensor([])
positions = gwr_model.getpos()
print('positions: ',len(positions))
for i in positions.keys():
	# print(torch.ones(positions[i].shape[0])*i) #(positions[i].shape)
	svmlabels = torch.cat((svmlabels,torch.ones(positions[i].shape[0])*i))
	inputvectors = torch.cat((inputvectors,positions[i]))
print(inputvectors.shape)
print(svmlabels.shape)

start_time1= time.time()
# nca = NeighborhoodComponentsAnalysis(random_state=42)
# nca.fit(inputvectors, svmlabels)
pca =  PCA(n_components=200)
pca.fit(inputvectors)
# lda = LinearDiscriminantAnalysis()
# lda.fit(inputvectors, svmlabels)
# clf = SVC(gamma='auto',kernel='rbf',degree=3)
# clf.fit(nca.transform(inputvectors), svmlabels)
# end_time1 = time.time()
clf = KNeighborsClassifier(n_neighbors=25,weights='distance', p=1.2)#,algorithm='brute')
clf.fit(pca.transform(inputvectors), svmlabels)
# clf = LinearDiscriminantAnalysis()
# clf.fit(inputvectors, svmlabels)
end_time1 = time.time()
print('time taken for nca and knn',end_time1 - start_time1)

start_time1= time.time()
if TestFC and TestGWR:
	# gwr_overall,gwr_categorical_acc,mh_overall,mh_cl_bycl,nodes_per_task = test_MultiNet(resnet_model,MultiNet, clf, test_loader, test_loader_size, number_of_tasks,num_classes_in_task,KNN_k,gwr=True,FC=True)
	gwr_overall,gwr_categorical_acc,mh_overall,mh_cl_bycl,nodes_per_task = test_MultiNet(resnet_model,MultiNet, clf, pca, test_loader, test_loader_size, number_of_tasks,num_classes_in_task,KNN_k,gwr=True,FC=True)
elif TestFC:
	mh_overall,mh_cl_bycl = test_MultiNet(resnet_model,MultiNet, clf, pca, test_loader, test_loader_size, number_of_tasks,num_classes_in_task,KNN_k,gwr=False,FC=True)
end_time1 = time.time()
print('average inference time: ', (end_time1- start_time1)/test_loader_size)

if TestGWR:
	print('GWR Parameters: ')
	print('act_thr: ', act_thr,' fir_thr: ', fir_thr, ' eps_b: ', eps_b, ' eps_n: ', eps_n,' tau_b: ', tau_b, ' tau_n: ', tau_n, ' gwr_epochs: ', gwr_epochs)
if TestFC:
	print('FC Parameters: ')
	print('Lr: ',learning_rate, 'gamma: ',gamma,' batch_size: ',batch_size,'step_size: ',step_size,' fc_epochs: ',fc_num_epochs)
print("Time taken fro the whole training : ",end_time_o - start_time_o)

if TestGWR:
	print('GWR nodes per task: ', nodes_per_task)
	print("GWR categorical accuracy: ", gwr_categorical_acc )
	print("GWR overall accuracy: ", gwr_overall)
if TestFC:
	print("MH class by class accuracy: ", mh_cl_bycl)
	print("MH overall accuracy: ", mh_overall)

# np.savez('trained_FC_model__full_dataset_1.pt', gwr_model)
# torch.save(model.state_dict(),'FC_model/model_dataset_full_1.pt')

