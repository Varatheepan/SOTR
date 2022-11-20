################################################
	# Kangarajah Sathursan
	# ksathursan1408@gmail.com
################################################
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################
	# Cumulative and Naive
################################################
def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:
			if transform:
				images =[]
				for image in inputs:
					img = transform(image.numpy())
					img = img.transpose(0,2).transpose(0,1)
					images.append(img)
				inputs = torch.stack(images)
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
		scheduler.step()
	return model

def test_model(model, criterion, optimizer, test_loader, test_loader_size, acc_file_name, transform = None):
	best_acc = 0.0
	model.eval()
	test_loss = 0.0
	corrects = 0
	number_of_correct_by_class = np.zeros(10, dtype = int)
	number_by_class = np.zeros(10, dtype = int)
	confusion_matrix = np.zeros((10, 10), dtype = int)
	for inputs, labels in test_loader:
		if transform:
			images =[]
			for image in inputs:
				img = transform(image.numpy())
				img = img.transpose(0,2).transpose(0,1)
				images.append(img)
			inputs = torch.stack(images)
		with torch.no_grad():
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)
			test_loss += loss.item() * inputs.size(0)
			corrects += torch.sum(preds == labels.data)
			for k in range(preds.size(0)):
				confusion_matrix[labels[k]][preds[k]] = confusion_matrix[labels[k]][preds[k]] + 1
				number_by_class[labels[k]] = number_by_class[labels[k]] + 1
				if preds[k] == labels[k]:
					number_of_correct_by_class[preds[k]] = number_of_correct_by_class[preds[k]] + 1
	
	print(number_of_correct_by_class)
	print(number_by_class)
	epoch_loss = test_loss / test_loader_size
	print(corrects.item())
	print(test_loader_size)
	epoch_acc = corrects.double() / test_loader_size
	print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
	# print(confusion_matrix)
	# if epoch_acc > best_acc:
	#   best_acc = epoch_acc
	#   best_model_wts = copy.deepcopy(model.state_dict())
	#   torch.save({'epoch': epoch,
	#               'model_state_dict': model.state_dict(),
	#               'optimizer_state_dict': optimizer.state_dict(),
	#               'loss': loss}, 'dumps/model/model_ewc.pth')
	# np.savez(acc_file_name, number_of_correct_by_class, number_by_class, confusion_matrix)
	return  number_of_correct_by_class, number_by_class, confusion_matrix

'''
################################################
	# Elastic Weight Consolidation
################################################
def train_ewc(model, criterion, optimizer, scheduler, train_loader, fisher_dict, optpar_dict, ewc_lamda, task_id, num_epochs, transform = None):
	scheduler.step()
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if task_id != 0:
					for task in range(task_id):
						for name, parameter in model.named_parameters():
							fisher = fisher_dict[task][name]
							optpar = optpar_dict[task][name]
							loss += (fisher * (optpar - parameter).pow(2)).sum() * ewc_lamda
				loss.backward()
				optimizer.step()
	return model

def task_update(model, criterion, optimizer, scheduler, train_loader, fisher_dict, optpar_dict, task_id, transform = None):
	scheduler.step()
	model.train()
	print('Updating fisher and parameters...')
	for inputs, labels in train_loader:
		inputs = inputs.to(device)
		labels = labels.to(device)
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
	# Elastic Weight Consolidation SH Version
################################################
def train_ewc(model, criterion, optimizer, scheduler, train_loader, fisher_dict, optpar_dict, ewc_lamda, task_id, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:	
			if transform:
				images =[]
				for image in inputs:
					img = transform(image.numpy())
					img = img.transpose(0,2).transpose(0,1)
					images.append(img)
				inputs = torch.stack(images)		
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if task_id != 0:
					for name, parameter in model.named_parameters():
						fisher = fisher_dict[name]
						optpar = optpar_dict[name]   
						if name != 'fc.weight' and name != 'fc.bias':    
							# loss += (fisher * (optpar - parameter).pow(2)).sum() * ewc_lamda
							loss += (fisher * (optpar - parameter).pow(2)).sum() * 0
						else:
							# fisher loss is calculated only for task 1 out of 2 tasks
							loss += (fisher[:task_id + 5] * (optpar[:task_id + 5] - parameter[: task_id + 5]).pow(2)).sum() * ewc_lamda	
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
		inputs, labels = data
		if transform:
			images =[]
			for image in inputs:
				img = transform(image.numpy())
				img = img.transpose(0,2).transpose(0,1)
				images.append(img)
			inputs = torch.stack(images)
		inputs = inputs.to(device)
		labels = labels.to(device)
		with torch.set_grad_enabled(True):
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			len_train_loader += len(inputs)
		if batch == 0:
			for name, parameter in model.named_parameters():
				fisher_dict[name] = (parameter.grad.data.clone().pow(2) * len(inputs))
				# optpar_dict[name] = parameter.data.clone()
				# fisher_dict[name] = torch.clamp(fisher_dict[name], min = -0.001, max = 0.001)
		else:
			for name, parameter in model.named_parameters():
				fisher_dict[name] += (parameter.grad.data.clone().pow(2) * len(inputs))       
	
	for name, parameter in model.named_parameters():    
		optpar_dict[name] = parameter.data.clone()
		fisher_dict[name] /= len_train_loader
		fisher_dict[name] = torch.clamp(fisher_dict[name], min = 0, max = 0.001)
	return fisher_dict, optpar_dict


'''


################################################
	# Elastic Weight Consolidation MH version
################################################
def train_ewc(model, criterion, optimizer, scheduler, train_loader, fisher_dict, optpar_dict, ewc_lamda, task_id, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:	
			if transform:
				images =[]
				for image in inputs:
					img = transform(image.numpy())
					img = img.transpose(0,2).transpose(0,1)
					images.append(img)
				inputs = torch.stack(images)		
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if task_id != 0:
					for name, parameter in model.named_parameters():
						if name == 'fc.weight' or name == 'fc.bias':
							continue
						fisher = fisher_dict[name]
						optpar = optpar_dict[name]                      
						loss += (fisher * (optpar - parameter).pow(2)).sum() * ewc_lamda
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
		inputs, labels = data
		if transform:
			images =[]
			for image in inputs:
				img = transform(image.numpy())
				img = img.transpose(0,2).transpose(0,1)
				images.append(img)
			inputs = torch.stack(images)
		inputs = inputs.to(device)
		labels = labels.to(device)
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
	for name, parameter in model.named_parameters():    
		optpar_dict[name] = parameter.data.clone()
		fisher_dict[name] /= len_train_loader
		fisher_dict[name] = torch.clamp(fisher_dict[name], min = -0.001, max = 0.001)
	# for name, parameter in model.named_parameters():
	# 	if name == 'fc.weight' or name == 'fc.bias':
	# 		mask = torch.zeros(fisher_dict[name].size())
	# 		mask[:task_id + 5] = 1		# 5 stands for number of classes per task
	# 		mask = mask.to(device)
	# 		fisher_dict[name] = fisher_dict[name] * mask
	return fisher_dict, optpar_dict

'''

















'''
################################################
	# Synaptic Intelligence
################################################
def train_si(model, criterion, optimizer, scheduler, train_loader, previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor, task_id, num_epochs, transform = None):
	model.train()
	for name, parameter in model.named_parameters():
		name = name.replace('.', '_')
		small_omega[name] = parameter.data.clone().zero_()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				if task_id == 0:
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
		scheduler.step()
	return model    

def surrogate_loss(model, previous_task_data):
	surr_loss = 0
	for name, parameter in model.named_parameters():
		if parameter.requires_grad:
			name = name.replace('.', '_')
			prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
			big_omega = previous_task_data['{}_big_omega'.format(name)]
			surr_loss += (big_omega * (parameter - prev_task_parameter)**2).sum()
	return surr_loss

def big_omega_update(model, previous_task_data, small_omega, epsilon, task_id):
	epsilon = torch.tensor(epsilon).cuda()
	for name, parameter in model.named_parameters():
		if parameter.requires_grad:
			name = name.replace('.', '_')
			current_task_parameter = parameter.detach().clone()
			if task_id == 0:
				big_omega = parameter.detach().clone().zero_()
				big_omega += small_omega[name].cuda() / torch.add((torch.zeros(current_task_parameter.shape))**2, epsilon).cuda()
			else:
				prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
				big_omega = previous_task_data['{}_big_omega'.format(name)]
				big_omega += small_omega[name].cuda() / torch.add((current_task_parameter - prev_task_parameter)**2, epsilon).cuda()
			previous_task_data.update({'{}_prev_task'.format(name): current_task_parameter})
			previous_task_data.update({'{}_big_omega'.format(name): big_omega})
'''
'''
################################################
	# Synaptic Intelligence Version 2
################################################
def train_si(model, criterion, optimizer, scheduler, train_loader, previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor, task_id, num_epochs, transform = None):
	model.train()
	for name, parameter in model.named_parameters():
		name = name.replace('.', '_')
		small_omega[name] = parameter.data.clone().zero_()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if task_id != 0:
					# print(scaling_factor * surrogate_loss(model, previous_task_data))
					loss += scaling_factor * surrogate_loss(model, previous_task_data)
				loss.backward()
				for name, parameter in model.named_parameters():
					# if parameter.requires_grad:
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
			surr_loss += (big_omega * (parameter - prev_task_parameter)**2).sum()
	return surr_loss

def big_omega_update(model, previous_task_data, small_omega, epsilon, task_id):
	epsilon = torch.tensor(epsilon).cuda()
	for name, parameter in model.named_parameters():
		if parameter.requires_grad:
			name = name.replace('.', '_')
			current_task_parameter = parameter.detach().clone()
			if task_id == 0:
				big_omega = parameter.detach().clone().zero_()
				big_omega += small_omega[name].cuda() / torch.add((torch.zeros(current_task_parameter.shape))**2, epsilon).cuda()
			else:
				prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
				big_omega = previous_task_data['{}_big_omega'.format(name)]
				big_omega += small_omega[name].cuda() / torch.add((current_task_parameter - prev_task_parameter)**2, epsilon).cuda()
			previous_task_data.update({'{}_prev_task'.format(name): current_task_parameter})
			previous_task_data.update({'{}_big_omega'.format(name): big_omega})


'''


################################################
	# Synaptic Intelligence SH Version
################################################
def train_si(model, criterion, optimizer, scheduler, train_loader, previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor, task_id, num_epochs, transform = None):
	for name, parameter in model.named_parameters():
		small_omega[name] = parameter.data.detach().clone().zero_()
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:
			if transform:
				images =[]
				for image in inputs:
					img = transform(image.numpy())
					img = img.transpose(0,2).transpose(0,1)
					images.append(img)
				inputs = torch.stack(images)
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)

				if task_id != 0:
					loss += scaling_factor * surrogate_loss(model, previous_task_data)

				loss.backward()
				optimizer.step()
				for name, parameter in model.named_parameters():
					small_omega[name].add_(-parameter.grad.detach().clone() * (parameter.detach() - previous_parameters[name]))
					previous_parameters[name] = parameter.detach().clone()					
		scheduler.step()
	return previous_parameters, previous_task_data, small_omega

def surrogate_loss(model, previous_task_data):
	surr_loss = 0
	for name, parameter in model.named_parameters():
		prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
		big_omega = previous_task_data['{}_big_omega'.format(name)]
		surr_loss += (big_omega * (parameter - prev_task_parameter)**2).sum()
	return surr_loss

def big_omega_update(model, previous_task_data, small_omega, epsilon, task_id):
	epsilon = torch.tensor(epsilon).cuda()
	for name, parameter in model.named_parameters():
		current_task_parameter = parameter.detach().clone()
		if task_id == 0:
			big_omega = parameter.detach().clone().zero_()
			big_omega += small_omega[name].cuda() / torch.add((torch.zeros(current_task_parameter.shape))**2, epsilon).cuda()
		else:
			prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
			big_omega = previous_task_data['{}_big_omega'.format(name)]
			big_omega += small_omega[name].cuda() / torch.add((current_task_parameter - prev_task_parameter)**2, epsilon).cuda()
		previous_task_data.update({'{}_prev_task'.format(name): current_task_parameter})
		previous_task_data.update({'{}_big_omega'.format(name): big_omega})
	return previous_task_data, small_omega


'''


################################################
	# Synaptic Intelligence MH Version 
################################################
def train_si(model, criterion, optimizer, scheduler, train_loader, previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor, task_id, num_epochs, transform = None):
	for name, parameter in model.named_parameters():
		small_omega[name] = parameter.data.detach().clone().zero_()
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:
			if transform:
				images =[]
				for image in inputs:
					img = transform(image.numpy())
					img = img.transpose(0,2).transpose(0,1)
					images.append(img)
				inputs = torch.stack(images)
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if task_id != 0:
					loss += scaling_factor * surrogate_loss(model, previous_task_data)
				loss.backward()
				optimizer.step()
				for name, parameter in model.named_parameters():
					small_omega[name].add_(-parameter.grad.detach().clone() * (parameter.detach() - previous_parameters[name]))
					previous_parameters[name] = parameter.detach().clone()					
		scheduler.step()
	return previous_parameters, previous_task_data, small_omega

def surrogate_loss(model, previous_task_data):
	surr_loss = 0
	for name, parameter in model.named_parameters():
		if name == 'fc.weight' or name == 'fc.bias':
			continue
		prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
		big_omega = previous_task_data['{}_big_omega'.format(name)]
		surr_loss += (big_omega * (parameter - prev_task_parameter)**2).sum()
	return surr_loss

def big_omega_update(model, previous_task_data, small_omega, epsilon, task_id):
	epsilon = torch.tensor(epsilon).cuda()
	for name, parameter in model.named_parameters():
		current_task_parameter = parameter.detach().clone()
		if task_id == 0:
			big_omega = parameter.detach().clone().zero_()
			big_omega += small_omega[name].cuda() / torch.add((torch.zeros(current_task_parameter.shape))**2, epsilon).cuda()
		else:
			prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
			big_omega = previous_task_data['{}_big_omega'.format(name)]
			big_omega += small_omega[name].cuda() / torch.add((current_task_parameter - prev_task_parameter)**2, epsilon).cuda()
		previous_task_data.update({'{}_prev_task'.format(name): current_task_parameter})
		previous_task_data.update({'{}_big_omega'.format(name): big_omega})
	return previous_task_data, small_omega


'''


################################################
	# Learning Without Forgetting
################################################
def soft_max_update(model, train_loader, soft_dict, transform = None):
	model.train()
	print('Collecting outputs for new task with old task parameters...')
	for inputs, labels, idx in train_loader:
		if transform:
			images =[]
			for image in inputs:
				img = transform(image.numpy())
				img = img.transpose(0,2).transpose(0,1)
				images.append(img)
			inputs = torch.stack(images)
		inputs = inputs.to(device)
		labels = labels.to(device)
		idx = idx.to(device)
		with torch.set_grad_enabled(False):
			outputs = model(inputs)
			for i in range(len(idx)):
				# print(F.softmax(outputs[i], dim = 0))
				soft_dict[idx[i].item()] = F.softmax(outputs[i], dim = 0)
	return soft_dict

def kl_divergence_loss(outputs, idx, soft_dict):
	klloss = nn.KLDivLoss(reduction = 'sum')
	new_out_distribution = {}
	kl_div_loss = 0
	for i in range(len(idx)):
		new_distribution = F.softmax(outputs[i], dim = 0)[:5]
		old_distribution = soft_dict[idx[i].item()][:5]
		# print(new_distribution)
		# print(old_distribution)
		# print(klloss(new_distribution.log(), old_distribution))
		kl_div_loss += klloss(new_distribution.log(), old_distribution)
	kl_div_loss /= len(idx)
	return kl_div_loss

def train_lwf(model, criterion, optimizer, scheduler, train_loader, soft_dict, lamda, task_id, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels, idx in train_loader:
			if transform:
				images =[]
				for image in inputs:
					img = transform(image.numpy())
					img = img.transpose(0,2).transpose(0,1)
					images.append(img)
				inputs = torch.stack(images)
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				for params in model.parameters():
					params.requires_grad = True
				if task_id != 0 and epoch == 1:
					for params in model.parameters():
						params.requires_grad = False
					for params in model.fc.parameters():
						params.requires_grad = True
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if task_id != 0 and epoch != 1:
					kl_loss = lamda * kl_divergence_loss(outputs, idx, soft_dict)
					loss += kl_loss
				loss.backward()
				
				if task_id != 0 and epoch == 1:
					for params in model.fc.parameters():
						# print(params.grad.data.clone())
						mask = torch.ones(params.grad.data.size())
						mask[:task_id] = 0
						mask = mask.to(device)
						params.grad.data = params.grad.data * mask
					# for params in model.fc.parameters():
						# print(params.grad.data.clone())
				optimizer.step()
		scheduler.step()
	return model

'''
################################################
	# CWR and CWR+
################################################
def train_cwr(model, criterion, optimizer, scheduler, train_loader, task_id, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			if task_id == 0:
				with torch.set_grad_enabled(True):
					outputs = model(inputs)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()
			else:
				with torch.set_grad_enabled(True):
					for params in model.conv1.parameters():
						params.requires_grad = False
					for params in model.conv2.parameters():
						params.requires_grad = False
					outputs = model(inputs)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()    
	return model

def new_fc_layer_model(model, weight_dict, bias_dict, scalars, number_of_tasks):
	new_fc_weight = []
	new_fc_bias = []
	for i in range(number_of_tasks):
		mean_w = weight_dict[i].mean()
		mean_b = bias_dict[i].mean()
		weight = (weight_dict[i][i]*scalars[i]).tolist()
		w = np.array(weight)
		bias = (bias_dict[i][i]*scalars[i]).tolist()
		weight = [(element - mean_w.item()) for element in weight]
		bias = bias - mean_b
		new_fc_weight.append(weight)
		new_fc_bias.append(bias)

	zero_weight = [0 for k in range(800)]
	zero_bias = 0
	for empty_index in range(10 - number_of_tasks):
		new_fc_weight.append(zero_weight)
		new_fc_bias.append(zero_bias)

	new_fc_weight = torch.FloatTensor(new_fc_weight)
	new_fc_bias = torch.FloatTensor(new_fc_bias)
	new_fc_weight.to(device)
	new_fc_bias.to(device)
	model.state_dict()['fc.weight'].copy_(new_fc_weight)
	model.state_dict()['fc.bias'].copy_(new_fc_bias)
	return model
'''

################################################
	# CWR and CWR+ V2
################################################
def train_cwr(model, criterion, optimizer, scheduler, train_loader, task_id, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for inputs, labels in train_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			if task_id == 0:
				with torch.set_grad_enabled(True):
					outputs = model(inputs)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()
			else:
				with torch.set_grad_enabled(True):
					for params in model.conv1.parameters():
						params.requires_grad = False
					for params in model.conv2.parameters():
						params.requires_grad = False
					outputs = model(inputs)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()    
	return model

# Function for CWR
# def new_fc_layer_model(model, weight_dict, bias_dict, scalars):
# 	new_fc_weight = torch.cat((weight_dict[0]*scalars[0], weight_dict[1]*scalars[1]), 0)
# 	new_fc_bias = torch.cat((bias_dict[0]*scalars[0], bias_dict[1]*scalars[1]), 0)
# 	new_fc_weight.to(device)
# 	new_fc_bias.to(device)
# 	model.state_dict()['fc.weight'].copy_(new_fc_weight)
# 	model.state_dict()['fc.bias'].copy_(new_fc_bias)
# 	return model

# Function for CWR+
def new_fc_layer_model(model, weight_dict, bias_dict):
	# new_fc_weight = torch.cat((weight_dict[0] - weight_dict[0].mean(), weight_dict[1] - weight_dict[1].mean()), 0)
	# new_fc_bias = torch.cat((bias_dict[0] - bias_dict[0].mean(), bias_dict[1] - bias_dict[1].mean()), 0)
	
	new_fc_weight = torch.cat((weight_dict[0], weight_dict[1]), 0)
	new_fc_bias = torch.cat((bias_dict[0], bias_dict[1]), 0)
	new_fc_weight = new_fc_weight - new_fc_weight.mean()
	new_fc_bias = new_fc_bias - new_fc_weight.mean()
	new_fc_weight.to(device)
	new_fc_bias.to(device)
	model.state_dict()['fc.weight'].copy_(new_fc_weight)
	model.state_dict()['fc.bias'].copy_(new_fc_bias)
	return model

################################################
	# Native replay
################################################
def buffer_indices_generator(replay_buffer_size, number_of_splits):
	indices = []
	quotient = replay_buffer_size // number_of_splits
	reminder = replay_buffer_size % number_of_splits
	for i in range(number_of_splits - reminder):
		indices.append(quotient)
	for i in range(reminder):
		indices.append(quotient + 1)
	return indices

def replay_data_generator(data, replay_data, replay_buffer_size, task_id):
	number_of_splits = task_id + 1
	new_indices = buffer_indices_generator(replay_buffer_size, number_of_splits)
	if task_id == 0:
		replay_data += data[:replay_buffer_size]
	else:
		old_indices = buffer_indices_generator(replay_buffer_size, number_of_splits - 1)
		removal_starting_index = 0
		for j in range(len(old_indices)):
			no_of_samples_removed = old_indices[j] - new_indices[j]
			removal_starting_index += new_indices[j]
			del replay_data[removal_starting_index : removal_starting_index + no_of_samples_removed]
		replay_data += data[:new_indices[-1]]
	return replay_data



























