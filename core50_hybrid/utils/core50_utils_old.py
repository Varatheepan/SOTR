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
import os
import copy
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import networkx as nx
import logging
import scipy.spatial.distance as sp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
################################################
	# Common
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

def test_model(model, criterion, optimizer, test_loader, test_loader_size, number_of_classes, transform = None):
	best_acc = 0.0
	model.eval()
	test_loss = 0.0
	corrects = 0
	number_of_correct_by_class = np.zeros((1, number_of_classes), dtype = int)
	number_by_class = np.zeros((1, number_of_classes), dtype = int)
	confusion_matrix = np.zeros((number_of_classes, number_of_classes), dtype = int)
	for image_names, labels in test_loader:
		with torch.no_grad():
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)
			test_loss += loss.item() * inputs.size(0)
			corrects += torch.sum(preds == labels.data)
			for k in range(preds.size(0)):
				confusion_matrix[labels[k]][preds[k]] = confusion_matrix[labels[k]][preds[k]] + 1
				number_by_class[0][labels[k]] = number_by_class[0][labels[k]] + 1
				if preds[k] == labels[k]:
					number_of_correct_by_class[0][preds[k]] = number_of_correct_by_class[0][preds[k]] + 1
	
	print(number_of_correct_by_class)
	print(number_by_class)
	epoch_loss = test_loss / test_loader_size
	print(corrects.item())
	print(test_loader_size)
	epoch_acc = corrects.double() / test_loader_size
	print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
	# if epoch_acc > best_acc:
	# 	best_acc = epoch_acc
	# 	best_model_wts = copy.deepcopy(model.state_dict())
	# 	torch.save({'epoch': epoch,
	# 				'model_state_dict': model.state_dict(),
	# 				'optimizer_state_dict': optimizer.state_dict(),
	# 				'loss': loss}, 'dumps/model_ewc.pth')
	return	number_of_correct_by_class, number_by_class, confusion_matrix

def train_model(model, criterion, optimizer, scheduler, train_loader,  num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
		scheduler.step()
	return model

'''
################################################
	# Elastic Weight Consolidation
################################################
def train_ewc(model, criterion, optimizer, scheduler, train_loader, fisher_dict, optpar_dict, ewc_lambda, task_id, num_epochs, transform = None):
	scheduler.step()
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if task_id != 1:
					for task in range(1 , task_id):
						for name, parameter in model.named_parameters():
							fisher = fisher_dict[task][name]
							optpar = optpar_dict[task][name]
							loss += (fisher * (optpar - parameter).pow(2)).sum() * ewc_lambda
				loss.backward()
				optimizer.step()
	return model

def task_update(model, criterion, optimizer, scheduler, train_loader, fisher_dict, optpar_dict, task_id, transform = None):
	scheduler.step()
	model.train()
	print('Updating fisher and parameters...')
	for image_names, labels in train_loader:
		inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
		with torch.set_grad_enabled(True):
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
	fisher_dict[task_id] = {}
	optpar_dict[task_id] = {}
	for name, parameter in model.named_parameters():
		optpar_dict[task_id][name] = parameter.data.clone()
		fisher_dict[task_id][name] = parameter.grad.data.clone().pow(2)
'''

################################################
	# Elastic Weight Consolidation Version 2
################################################
def train_ewc(model, criterion, optimizer, scheduler, train_loader, fisher_dict, optpar_dict, ewc_lambda, task_id, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if task_id != 1:
					for name, parameter in model.named_parameters():
						fisher = fisher_dict[name]
						optpar = optpar_dict[name]							
						# loss += (fisher * (optpar - parameter).pow(2)).sum() * ewc_lambda
						if name != 'fc.weight' and name != 'fc.bias':
							loss += (fisher * (optpar - parameter).pow(2)).sum() * ewc_lambda
						else:
							parameter_ = parameter[:optpar.shape[0]]
							loss += (fisher * (optpar - parameter_).pow(2)).sum() * ewc_lambda
				loss.backward()
				optimizer.step()
		scheduler.step()
	return model

def task_update(model, criterion, optimizer, scheduler, train_loader, fisher_dict, optpar_dict, task_id, transform = None):
	model.train()
	print('Updating fisher and parameters...')
	len_train_loader = 0
	fisher_dict = {}
	optpar_dict = {}
	for batch, data in enumerate(train_loader):
		image_names, labels = data
		inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
		optimizer.zero_grad()
		with torch.set_grad_enabled(True):
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			len_train_loader += len(inputs)
		if batch == 0:
			for name, parameter in model.named_parameters():
				fisher_dict[name] = (parameter.grad.data.clone().pow(2) * len(inputs))
		else:
			for name, parameter in model.named_parameters():
				fisher_dict[name] += (parameter.grad.data.clone().pow(2) * len(inputs))	
		# print(len_train_loader)		
	for name, parameter in model.named_parameters():
		optpar_dict[name] = parameter.data.clone()
		fisher_dict[name] /= len_train_loader
		fisher_dict[name] = torch.clamp(fisher_dict[name], min = -0.0001, max = 0.0001)
	print(fisher_dict['fc.weight'])	
	return fisher_dict, optpar_dict

def new_fc_layer_model(model, num_ftrs, task_id):
	old_weight = copy.deepcopy(model.fc.weight.data[...])
	old_bias = copy.deepcopy(model.fc.bias.data[...])

	for name, parameter in model.named_parameters():
		if name == 'fc.weight':
			print(name)
			print(parameter.data.clone())

	model.fc = nn.Linear(num_ftrs, task_id)
	model = model.to(device)
	new_weight = copy.deepcopy(model.fc.weight.data[...])
	new_bias = copy.deepcopy(model.fc.bias.data[...])
	new_fc_weight = []
	new_fc_bias = []
	for i in range(task_id - 1):
		weight = (old_weight[i]).tolist()
		bias = (old_bias[i]).tolist()
		new_fc_weight.append(weight)
		new_fc_bias.append(bias)
	weight = (new_weight[task_id - 1]).tolist()
	bias = (new_bias[task_id - 1]).tolist()
	new_fc_weight.append(weight)
	new_fc_bias.append(bias)	
	new_fc_weight = torch.FloatTensor(new_fc_weight)
	new_fc_bias = torch.FloatTensor(new_fc_bias)
	new_fc_weight.to(device)
	new_fc_bias.to(device)
	model.state_dict()['fc.weight'].copy_(new_fc_weight)
	model.state_dict()['fc.bias'].copy_(new_fc_bias)
	return model

################################################
	# Learning Without Forgetting
################################################
def soft_max_update(model, train_loader, soft_dict, transform = None):
	model.train()
	for image_names, labels, idx in train_loader:
		inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
		inputs = inputs.to(device)
		labels = labels.to(device)
		idx = idx.to(device)
		with torch.set_grad_enabled(False):
			outputs = model(inputs)
			for i in range(len(labels)):
				soft_dict[idx[i].item()] = F.softmax(outputs[i], dim = 0)
	return soft_dict

def kl_divergence_loss(outputs, idx, soft_dict):
	klloss = nn.KLDivLoss(reduction = 'sum')
	new_out_distribution = {}
	kl_div_loss = 0
	for i in range(len(idx)):
		new_distribution = F.softmax(outputs[i], dim = 0)[:5]
		old_distribution = soft_dict[idx[i].item()][:5]
		kl_div_loss += klloss(new_distribution.log(), old_distribution)
	kl_div_loss /= len(idx)
	return kl_div_loss

def train_lwf(model, criterion, optimizer, scheduler, train_loader, 
				soft_dict, lamda, task_id, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels, idx in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			inputs = inputs.to(device)
			labels = labels.to(device)
			idx = idx.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				for params in model.parameters():
					params.requires_grad = True
				if task_id != 1 and epoch == 1:
					for params in model.parameters():
						params.requires_grad = False
					for params in model.fc.parameters():
						params.requires_grad = True
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if task_id != 1 and epoch != 1:
					kl_loss = lamda * kl_divergence_loss(outputs, idx, soft_dict)
					loss += kl_loss
				loss.backward()
				
				if task_id != 1 and epoch == 1:
					# print(task_id)
					# print('DDDDDDDDDDDDDDDDDDDDDD')
					for params in model.fc.parameters():
						# print(params.grad.data.clone())
						mask = torch.ones(params.grad.data.size())
						mask[:task_id - 1] = 0
						mask = mask.to(device)
						params.grad.data = params.grad.data * mask
					# for params in model.fc.parameters():
						# print(params.grad.data.clone())
				optimizer.step()
		scheduler.step()
	return model





















'''
################################################
	# Learning Without Forgetting (OLD)
################################################
def soft_max_update(model, train_loader, soft_dict, transform = None):
	model.train()
	print('Collecting outputs for new task with old task parameters...')
	for idx, data in enumerate(train_loader):
		image_names, labels = data
		inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
		inputs = inputs.to(device)
		labels = labels.to(device)
		with torch.set_grad_enabled(False):
			outputs = model(inputs)
			soft_dict[idx] = F.softmax(outputs, dim = 1)
			# print(soft_dict[idx])
	return soft_dict

def kl_divergence_loss(new_output, idx, soft_dict):
	klloss = nn.KLDivLoss(reduction = 'sum')
	# kl_div_loss = klloss(soft_dict[idx].log() , new_output) / new_output.shape[0]
	kl_div_loss = klloss(new_output.log(), soft_dict[idx]) / new_output.shape[0]
	return kl_div_loss

def train_lwf(model, criterion, optimizer, scheduler, train_loader, soft_dict, lamda_list, task_id, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for idx, data in enumerate(train_loader):
			image_names, labels = data
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				if task_id != 1 and epoch == 1:
					for params in model.parameters():
						params.requires_grad = False
					for params in model.fc.parameters():
						params.requires_grad = True
				outputs = model(inputs)
				new_output = F.softmax(outputs, dim = 1)
				loss = criterion(outputs, labels)
				if task_id != 1:
					lamda = lamda_list[task_id - 1]
					kl_loss = lamda * kl_divergence_loss(new_output, idx, soft_dict)
					loss += kl_loss
				loss.backward()
				if task_id != 1 and epoch == 1:
					for params in model.fc.parameters():
						# print(params.grad.data.clone())
						mask = torch.ones(params.grad.data.size())
						mask[:task_id - 1] = 0
						mask = mask.to(device)
						params.grad.data = params.grad.data * mask
					# for params in model.fc.parameters():
						# print(params.grad.data.clone())
				optimizer.step()
		scheduler.step()
	return model

'''
































################################################
	# Synaptic Intelligence
################################################
def train_si(model, criterion, optimizer, scheduler, train_loader, previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor, task_id, num_epochs, transform = None):
	scheduler.step()
	model.train()
	for name, parameter in model.named_parameters():
		name = name.replace('.', '_')
		small_omega[name] = parameter.data.clone().zero_()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				if task_id == 1:
					loss = criterion(outputs, labels)
				else:
					loss = criterion(outputs, labels) + scaling_factor * surrogate_loss(model, previous_task_data)
				loss.backward()
				for name, parameter in model.named_parameters():
					if parameter.requires_grad:
						name = name.replace('.', '_')
						small_omega[name].add_(-parameter.grad * (parameter.detach() - previous_parameters[name]))
						previous_parameters[name] = parameter.detach().clone()
				optimizer.step()
	return model	

def surrogate_loss(model, previous_task_data):
	surr_loss = 0
	for name, parameter in model.named_parameters():
		if parameter.requires_grad:
			name = name.replace('.', '_')
			prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
			big_omega = previous_task_data['{}_big_omega'.format(name)]
			surr_loss += (big_omega ** (parameter - prev_task_parameter)**2).sum()
	return surr_loss

def big_omega_update(model, previous_task_data, small_omega, epsilon, task_id):
	epsilon = torch.tensor(epsilon).cuda()
	for name, parameter in model.named_parameters():
		if parameter.requires_grad:
			name = name.replace('.', '_')
			current_task_parameter = parameter.detach().clone()
			if task_id == 1:
				big_omega = parameter.detach().clone().zero_()
				big_omega += small_omega[name].cuda() / torch.add((torch.zeros(current_task_parameter.shape))**2, epsilon).cuda()
			else:
				prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
				big_omega = previous_task_data['{}_big_omega'.format(name)]
				big_omega += small_omega[name].cuda() / torch.add((current_task_parameter - prev_task_parameter)**2, epsilon).cuda()
			previous_task_data.update({'{}_prev_task'.format(name): current_task_parameter})
			previous_task_data.update({'{}_big_omega'.format(name): big_omega})





















################################################
	# GWR
################################################
class Multi_fcNet(nn.Module):
    def __init__(self):
        super(Multi_fcNet, self).__init__()
        self.multi_fcs = nn.ModuleDict({})		# Create a disctonary of multiple parralley connected fc layers

    def forward(self, x, fc_index):
        x = self.multi_fcs[str(fc_index)](x)
        return x

def train_gwr_fc(model, gwr_model, criterion, optimizer, scheduler, train_loader,
					num_classes_in_task, task_id, gwr_epochs, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			fc_labels = labels % num_classes_in_task
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, fc_labels)
				loss.backward()
				optimizer.step()
		scheduler.step()


		optimizer.zero_grad()		# feature extraction
		with torch.set_grad_enabled(False):
			feature_model = nn.Sequential(*list(model.children())[:-1])		# removes last fc layer keeping average pooloing layer
			features = torch.tensor([]).to(device)
			gwr_labels = torch.tensor([]).to(device)
			for image_names, labels in train_loader:
				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
				features_batch = feature_model(inputs)

				features_batch = features_batch.view(features_batch.data.shape[0],512)
				# print(features_batch.data.shape)
				gwr_labels_batch = labels // num_classes_in_task
				features = torch.cat((features, features_batch))
				gwr_labels = torch.cat((gwr_labels, gwr_labels_batch.float()))
			# print(features.data.shape)
			features = features.cpu().numpy()
			gwr_labels = gwr_labels.cpu().numpy()

			if(task_id == 0):
				graph_gwr1 = gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs)	# GWR initiation
			else:
				graph_gwr1 = gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs, warm_start = True)	# continuing
			# graph_gwr = g.train(output, n_epochs=epochs)
			# Xg = g.get_positions()
			number_of_clusters1 = nx.number_connected_components(graph_gwr1)	# number of dintinct clusters without any connections
			# print('number of clusters: ',number_of_clusters1)
			num_nodes1 = graph_gwr1.number_of_nodes()		# currently existing no of nodes end of training
			print('number of nodes: ',num_nodes1)
				






	return model


	# labels = labels.long()%num_classes_in_task
	# # print(labels)
	# for epoch in range(1, num_epochs + 1):
	# 	print('classifier epoch----- : ', epoch)			
	# 	feature_inputs = feature_inputs.to(device)
	# 	labels = labels.to(device)
	# 	optimizer.zero_grad()
	# 	with torch.set_grad_enabled(True):
	# 		outputs = model(feature_inputs,task_id)
	# 		loss = criterion(outputs, labels)
	# 		loss.backward()
	# 		optimizer.step()
	# 	scheduler.step()
	# return model


def train_model(model, criterion, optimizer, scheduler, train_loader,  num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
		scheduler.step()
	return model





## Firing update changed as original
class gwr_network():

	'''
	Growing When Required (GWR) Neural Gas, after [1]. Constitutes the base class
	for the Online Semi-supervised (OSS) GWR.

	[1] Parisi, G. I., Tani, J., Weber, C., & Wermter, S. (2017).
	Emergence of multimodal action representations from neural network
	self-organization. Cognitive Systems Research, 43, 208-221.
	'''

	def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
				 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, alpha_b = 1.05,
				 alpha_n = 1.05, h_0 = 1, sti_s = 1,
				 lab_thr = 0.5, max_age = 100, max_size = 100,
				 random_state = None):
		self.act_thr  = act_thr
		self.fir_thr  = fir_thr
		self.eps_b    = eps_b
		self.eps_n    = eps_n
		self.tau_b    = tau_b
		self.tau_n    = tau_n
		self.alpha_b    = alpha_b
		self.alpha_n    = alpha_n
		self.h_0   = h_0
		self.sti_s   = sti_s
		self.lab_thr  = lab_thr
		self.max_age  = max_age
		self.max_size = max_size
		self.num_changes = 0
		if random_state is not None:
			np.random.seed(random_state)

	def _initialize(self, X, Y):

		logging.info('Initializing the neural gas.')
		self.G = nx.Graph()
		# TODO: initialize empty labels?
		draw = np.random.choice(X.shape[0], size=2, replace=False)
		# print(X[draw[0]].shape)                                   #vara
		# print(draw)                                               #vara
		self.G.add_nodes_from([(0,{'pos' : X[draw[0],:],'fir' : self.sti_s, 'n_best' : 0, 'label' : Y[draw[0]], 'best_act' : 1})])      #vara
		self.G.add_nodes_from([(1,{'pos' : X[draw[1],:],'fir' : self.sti_s, 'n_best' : 0, 'label' : Y[draw[1]], 'best_act' : 1})])      #vara
		#print(self.G.number_of_nodes())                            #vara
		#print(nx.get_node_attributes(self.G, 'pos').values())      #vara
		#print(self.G.nodes[0])                                     #vara

	def get_positions(self):
		pos = np.array(list(nx.get_node_attributes(self.G, 'pos').values()))    #vara
		return pos

	def _get_best_matching(self, x):
		pos = self.get_positions()
		#np.concatenate(pos,axis = 0)               #vara
		#print(pos)           
		dist = sp.cdist(x, pos, metric='euclidean')
		sorted_dist = np.argsort(dist)
		#print('sroted dist' , sorted_dist)         #vara
		b = sorted_dist[0,0]
		s = sorted_dist[0,1]

		self.G.nodes[b]['n_best'] += 1

		return b, s


	def _get_activation(self, x, b):
		p = self.G.nodes[b]['pos'][np.newaxis,:]
		dist = sp.cdist(x, p, metric='euclidean')[0,0]
		# print('x ',x)
		# print('p ',p)
		# print('dist ',dist)
		act = np.exp(-dist)
		return act


	def _make_link(self, b, s):
		self.G.add_edge(b,s,age = 0)            #vara
		#print('edges ',self.G.edges([0]))      #vara


	def _add_node(self, x, y, b, s):
		r = max(self.G.nodes()) + 1
		pos_r = 0.5 * (x + self.G.nodes[b]['pos'])
		dist = sp.cdist(x, pos_r, metric='euclidean')[0,0]
		act = np.exp(-dist)
		pos_r = pos_r[0,:]
		self.G.add_nodes_from([(r, {'pos' : pos_r, 'fir' : self.sti_s, 'n_best' : 0, 'label' : y,'best_act' : act})])    #vara
		self.G.remove_edge(b,s)
		self.G.add_edge(r, b, age = 0)          #vara
		self.G.add_edge(r, s, age = 0)          #vara
		return r


	def _update_network(self, x, b):
		dpos_b = self.eps_b * self.G.nodes[b]['fir']*(x - self.G.nodes[b]['pos'])
		self.G.nodes[b]['pos'] = self.G.nodes[b]['pos'] + dpos_b[0,:]

		neighbors = self.G.neighbors(b)
		for n in neighbors:
			# update the position of the neighbors
			dpos_n = self.eps_n * self.G.nodes[n]['fir'] * (
					 x - self.G.nodes[n]['pos'])
			self.G.nodes[n]['pos'] = self.G.nodes[n]['pos'] + dpos_n[0,:]

			# increase the age of all edges connected to b
			#print('age ', self.G.edges[b,n]['age'])        #vara
			self.G.edges[b,n]['age'] += 1                   #vara


	def _update_firing(self, b):
		# dfir_b = self.tau_b * self.kappa*(1-self.G.nodes[b]['fir']) - self.tau_b        #different
		self.G.nodes[b]['fir'] = self.h_0 - (self.sti_s/self.alpha_b)*\
			(1-np.exp(-self.alpha_b*self.G.nodes[b]['n_best']/self.tau_b))          #vara
		
		# if(self.G.nodes[b]['fir']<0):         #vara
		#     print("negtive problem")
		# self.G.nodes[b]['fir'] = np.clip(self.G.nodes[b]['fir'],0,1) #max(0,self.G.nodes[b]['fir']);
		#print('fir ',self.G.nodes[b]['fir'])   #vara
		#print(self.G.nodes)                    #vara

		neighbors = self.G.neighbors(b)
		for n in neighbors:
			self.G.nodes[n]['fir'] = self.h_0 - (self.sti_s/self.alpha_n)*\
				(1-np.exp(-self.alpha_n*self.G.nodes[n]['n_best']/self.tau_n))
			# self.G.nodes[n]['fir'] = np.clip(self.G.nodes[n]['fir'],0,1)


	def _remove_old_edges(self):
		for e in self.G.edges():
			if self.G[e[0]][e[1]]['age'] > self.max_age:
				self.G.remove_edge(*e)
		self.G.remove_nodes_from(list(nx.isolates(self.G)))
		# for node in self.G.nodes():
		#     if len(self.G.edges(node)) == 0:
		#         logging.debug('Removing node %s', str(node))
		#         self.G.remove_node(node)

	def _check_stopping_criterion(self):
		# TODO: implement this
		pass

	def _training_step(self, x, y):
		# TODO: do not recompute all positions at every iteration
		b, s = self._get_best_matching(x)
		#print('best match ',b,s)            #vara
		self._make_link(b, s)
		act = self._get_activation(x, b)
		fir = self.G.nodes[b]['fir']
		logging.debug('Training step - best matching: %s, %s \n'
					  'Network activation: %s \n'
					  'Firing: %s', str(b), str(s), str(np.round(act,3)),
					  str(np.round(fir,3)))
		if act < self.act_thr and fir < self.fir_thr \
			and len(self.G.nodes()) < self.max_size:
			r = self._add_node(x, y, b, s)
			logging.debug('GENERATE NODE %s', self.G.nodes[r])
		else:
			self._update_network(x, b)
			if ((act > self.G.nodes[b]['best_act']) and (int(self.G.nodes[b]['label']) != int(y))):
				self.G.nodes[b]['label'] = y
				self.num_changes += 1

		self._update_firing(b)
		self._remove_old_edges()


	def train(self, X, Y, n_epochs=20, warm_start = False):
		if not warm_start:
			self._initialize(X,Y)
		for n in range(n_epochs):
			print('gwr epoch: ',n)
			logging.info('>>> Training epoch %s', str(n))
			for i in range(X.shape[0]):
				x = X[i,np.newaxis]
				y = Y[i]
				self._training_step(x,y)
				self._check_stopping_criterion()
		#nx.draw(self.G)        #vara
		#plt.show()
		logging.info('Training ended - Network size: %s', len(self.G.nodes())) 
		print('num_changes: ', self.num_changes) 
		self.num_changes = 0    
		return self.G

	def test(self, X, Y,num_tasks):                   #vara
		num_correct = 0
		class_by_class = np.zeros(num_tasks)
		class_by_class_pred = np.zeros(num_tasks)
		for i in range(X.shape[0]):
			x = X[i,np.newaxis]
			y = Y[i]
			class_by_class[int(y)] += 1
			b,s = self._get_best_matching(x)
			act = self._get_activation(x, b)
			# print('activation : ', act)
			y_pred = self.G.nodes[b]['label']
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class

	def _test_best_matching(self, x):
		pos = self.get_positions()
		#np.concatenate(pos,axis = 0)               #vara
		#print(pos)           
		dist = sp.cdist(x, pos, metric='euclidean')
		sorted_dist = np.argsort(dist)
		#print('sroted dist' , sorted_dist)         #vara
		# b = sorted_dist[0,0]
		# s = sorted_dist[0,1]
		return sorted_dist[0,:10]

	def KNearest_test(self, X, Y):                   #vara
		num_correct = 0
		class_by_class = np.zeros(10)
		class_by_class_pred = np.zeros(10)
		for i in range(X.shape[0]):
			x = X[i,np.newaxis]
			y = Y[i]
			class_by_class[int(y)] += 1
			best_matches = self._test_best_matching(x)
			votes = np.zeros(10)
			for j in best_matches:
				votes[int(self.G.nodes[j]['label'])] += 1
			y_pred = np.argsort(votes)[-1]
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class

	def choose_task(self, X, num_tasks):
		pred_class = np.zeros(len(X))
		for i in range(X.shape[0]):
			x = X[i,np.newaxis]
			# y = Y[i]
			best_matches = self._test_best_matching(x)
			votes = np.zeros(num_tasks)
			for j in best_matches:
				votes[int(self.G.nodes[j]['label'])] += 1
			y_pred = np.argsort(votes)[-1]
			pred_class[i] = y_pred
		return pred_class

	def nodes_per_task(self,num_tasks):
		tasks = np.zeros(num_tasks)
		for node in range(self.G.number_of_nodes()):
			tasks[int(self.G.nodes[node]['label'])] += 1
		return tasks