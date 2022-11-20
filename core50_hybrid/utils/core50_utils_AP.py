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
import random
import time

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
	return model	
'''
def surrogate_loss(model, previous_task_data):
	surr_loss = 0
	for name, parameter in model.named_parameters():
		if parameter.requires_grad:	
			name = name.replace('.', '_')
			prev_task_parameter = previous_task_data['{}_prev_task'.format(name)]
			big_omega = previous_task_data['{}_big_omega'.format(name)]
			surr_loss += (torch.mul(big_omega, (parameter - prev_task_parameter)**2)).sum()
	# print('surr_loss: ', surr_loss)
	return surr_loss
'''
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

'''
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

def big_omega_update(model, previous_task_data, small_omega, epsilon, task_id,si_parameters):
	epsilon = torch.tensor(epsilon).cuda()
	print(epsilon)
	for name, parameter in model.named_parameters():
		if name in si_parameters:
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






################################################
	# GWR
################################################
class Multi_fcNet(nn.Module):
	def __init__(self):
		super(Multi_fcNet, self).__init__()
		self.multi_fcs = nn.ModuleDict({})		# Create a dictionary of multiple parralley connected fc layers

	def forward(self, x, fc_index):
		x = self.multi_fcs[str(fc_index)](x)
		return x


def train_gwr(model, gwr_model,train_loader,num_classes_in_task, task_id, gwr_epochs, num_epochs, 
				gwr_imgs_skip, gwr_trained, individual_class, transform = None):

	feature_model = nn.Sequential(*list(model.children())[:-1])		# removes last fc layer keeping average pooloing layer
	feature_model.eval()
	# optimizer.zero_grad()		# feature extraction
	if not gwr_trained:
		with torch.set_grad_enabled(False):
			features = torch.tensor([]).to(device)
			gwr_labels = torch.tensor([]).to(device)
			for image_names, labels in train_loader:
				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
				features_batch = feature_model(inputs)

				features_batch = features_batch.view(features_batch.data.shape[0],512)
				if not individual_class:
					gwr_labels_batch = labels // num_classes_in_task
				else:
					gwr_labels_batch = labels
				features = torch.cat((features, features_batch))
				gwr_labels = torch.cat((gwr_labels, gwr_labels_batch.float()))
			features = features.cpu()[::gwr_imgs_skip]		#[::gwr_imgs_skip]
			gwr_labels = gwr_labels.cpu()[::gwr_imgs_skip]	#[::gwr_imgs_skip]

			if(task_id == 0):
				gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs)	# GWR initiation
			else:
				gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs, warm_start = True)	# continuing
	else:
		print('pretrained gwr given')
	num_nodes1 = gwr_model.get_num_nodes()		# currently existing no of nodes end of training
	print('number of nodes:', num_nodes1)
				
	return model, gwr_model

# def train_gwr_fc(model, gwr_model, criterion, optimizer,tuned_optimizer, scheduler, tuned_scheduler, train_loader,
# 					num_classes_in_task, task_id, gwr_epochs, num_epochs, gwr_imgs_skip, gwr_trained, replay_fc, 
# 					replay_samples = None, samples_per_task = None, transform = None):
# 	model.train()
# 	# model.eval()
# 	if replay_fc:
# 		i = 0
# 		random.shuffle(replay_samples)
# 		sample_set_size = len(replay_samples) - 1
# 		task_classes = range(task_id*num_classes_in_task,(task_id+1)*num_classes_in_task)
# 		for epoch in range(1, num_epochs + 1):						#vara
# 			print('Epoch : ', epoch)
# 			for image_names, labels in train_loader:
# 				# print(type(image_names))
# 				# print(labels)
# 				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
# 				# print(labels)
# 				if task_id != 0:
# 					sample_size = samples_per_task 
# 					if(i + sample_size > sample_set_size ):
# 						sample_size = sample_set_size - i
# 					replay_names= tuple(replay_samples[i+j][0] for j in range(sample_size))
# 					replay_labels = torch.tensor([replay_samples[i+j][1] for j in range(sample_size)])
# 					# print(replay_names)
# 					replay_data, replay_labels = read_load_inputs_labels(replay_names, replay_labels, transform = transform)
# 					outputs = model(inputs)
# 					replay_outputs = model(replay_data)
# 					loss = criterion(outputs, labels % num_classes_in_task)
# 					replay_loss = criterion(replay_outputs, replay_labels % num_classes_in_task)
# 					loss.backward(retain_graph=True)
# 					optimizer.step()
# 					replay_loss.backward()
# 					tuned_optimizer.step()
					


# 					i = i + sample_size
# 					if (i >= sample_set_size):
# 						i = 0
# 						random.shuffle(replay_samples)		
# 				else:
# 					outputs = model(inputs)
# 					# print('outputs: ',outputs)
# 					loss = criterion(outputs, labels)
# 					loss.backward()
# 					optimizer.step()
# 					tuned_optimizer.step()
# 		optimizer.step()
# 		tuned_optimizer.step()

# 	else:	
# 		for epoch in range(1, num_epochs + 1):						#vara
# 			print('Epoch : ', epoch)

# 			for image_names, labels in train_loader:
# 				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
# 				fc_labels = labels % num_classes_in_task
# 				optimizer.zero_grad()
# 				with torch.set_grad_enabled(True):
# 					outputs = model(inputs)
# 					loss = criterion(outputs, fc_labels)
# 					loss.backward()	
# 					optimizer.step()
# 					# tuned_optimizer.step()
# 			scheduler.step()
# 			# tuned_scheduler.step()

# 	if not gwr_trained:
# 		feature_model = nn.Sequential(*list(model.children())[:-1])		# removes last fc layer keeping average pooloing layer
# 		feature_model.eval()
# 		optimizer.zero_grad()		# feature extraction
# 		with torch.set_grad_enabled(False):
# 			features = torch.tensor([]).to(device)
# 			gwr_labels = torch.tensor([]).to(device)
# 			for image_names, labels in train_loader:
# 				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
# 				features_batch = feature_model(inputs)

# 				features_batch = features_batch.view(features_batch.data.shape[0],512)
# 				gwr_labels_batch = labels // num_classes_in_task
# 				features = torch.cat((features, features_batch))
# 				gwr_labels = torch.cat((gwr_labels, gwr_labels_batch.float()))
# 			features = features.cpu()[::gwr_imgs_skip]
# 			gwr_labels = gwr_labels.cpu()[::gwr_imgs_skip]

# 			if(task_id == 0):
# 				gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs)	# GWR initiation
# 			else:
# 				gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs, warm_start = True)	# continuing

# 		num_nodes1 = gwr_model.get_num_nodes()		# currently existing no of nodes end of training
# 	else:
# 		print('pre-trained gwr given')
				
# 	return model, gwr_model


def train_cnn_gwr_fc(model, gwr_model, criterion, optimizer,tuned_optimizer, scheduler, tuned_scheduler, train_loader,
					num_classes_in_task, task_id, gwr_epochs, num_epochs, gwr_imgs_skip, gwr_trained, transform = None):
	model.train()
	# model.eval()
	for epoch in range(1, num_epochs + 1):						#vara
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
				# tuned_optimizer.step()
		scheduler.step()
		# tuned_scheduler.step()

	if not gwr_trained:
		feature_model = nn.Sequential(*list(model.children())[:-1])		# removes last fc layer keeping average pooloing layer
		feature_model.eval()
		optimizer.zero_grad()		# feature extraction
		with torch.set_grad_enabled(False):
			features = torch.tensor([]).to(device)
			gwr_labels = torch.tensor([]).to(device)
			for image_names, labels in train_loader:
				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
				features_batch = feature_model(inputs)

				features_batch = features_batch.view(features_batch.data.shape[0],512)
				gwr_labels_batch = labels // num_classes_in_task
				features = torch.cat((features, features_batch))
				gwr_labels = torch.cat((gwr_labels, gwr_labels_batch.float()))
			features = features.cpu()[::gwr_imgs_skip]
			gwr_labels = gwr_labels.cpu()[::gwr_imgs_skip]

			if(task_id == 0):
				gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs)	# GWR initiation
			else:
				gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs, warm_start = True)	# continuing

		num_nodes1 = gwr_model.get_num_nodes()		# currently existing no of nodes end of training
	else:
		print('pre-trained gwr given')
				
	return model, gwr_model


# def train_gwr_fc(model, gwr_model, criterion, optimizer,tuned_optimizer, scheduler, tuned_scheduler, train_loader,
# 					num_classes_in_task, task_id, gwr_epochs, num_epochs, gwr_imgs_skip, gwr_trained, transform = None):
# 	model.train()
# 	# model.eval()
# 	for epoch in range(1, num_epochs + 1):						#vara
# 		print('Epoch : ', epoch)

# 		for image_names, labels in train_loader:
# 			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
# 			fc_labels = labels % num_classes_in_task
# 			optimizer.zero_grad()
# 			with torch.set_grad_enabled(True):
# 				outputs = model(inputs)
# 				loss = criterion(outputs, fc_labels)
# 				loss.backward()	
# 				optimizer.step()
# 				# tuned_optimizer.step()
# 		scheduler.step()
# 		# tuned_scheduler.step()

# 	if not gwr_trained:
# 		feature_model = nn.Sequential(*list(model.children())[:-1])		# removes last fc layer keeping average pooloing layer
# 		feature_model.eval()
# 		optimizer.zero_grad()		# feature extraction
# 		with torch.set_grad_enabled(False):
# 			features = torch.tensor([]).to(device)
# 			gwr_labels = torch.tensor([]).to(device)
# 			for image_names, labels in train_loader:
# 				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
# 				features_batch = feature_model(inputs)

# 				features_batch = features_batch.view(features_batch.data.shape[0],512)
# 				gwr_labels_batch = labels // num_classes_in_task
# 				features = torch.cat((features, features_batch))
# 				gwr_labels = torch.cat((gwr_labels, gwr_labels_batch.float()))
# 			features = features.cpu()[::gwr_imgs_skip]
# 			gwr_labels = gwr_labels.cpu()[::gwr_imgs_skip]

# 			if(task_id == 0):
# 				gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs)	# GWR initiation
# 			else:
# 				gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs, warm_start = True)	# continuing

# 		num_nodes1 = gwr_model.get_num_nodes()		# currently existing no of nodes end of training
# 	else:
# 		print('pre-trained gwr given')
				
# 	return model, gwr_model


def train_gwr_si(model, task_cnn_model, gwr_model, criterion, optimizer,tuned_optimizer, scheduler, tuned_scheduler, train_loader, train_loader1,
				previous_parameters, previous_task_data, small_omega, epsilon, scaling_factor, tuned_parameter,
				num_classes_in_task, task_id, gwr_epochs, num_epochs, gwr_imgs_skip, gwr_trained = False, transform = None):

	model.train()
	for name, parameter in model.named_parameters():
		if name in tuned_parameter:
		  name = name.replace('.', '_')
		  small_omega[name] = parameter.data.clone().zero_()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			fc_labels = labels % num_classes_in_task
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
			if task_id != 0:
				tuned_scheduler.step()

	# feature_model = nn.Sequential(*list(model.children())[:-1])		# removes last fc layer keeping average pooloing layer
	if not gwr_trained:
		feature_model = task_cnn_model
		feature_model.eval()
		optimizer.zero_grad()	
		tuned_optimizer.zero_grad()	# feature extraction
		
		with torch.set_grad_enabled(False):
			features = torch.tensor([]).to(device)
			gwr_labels = torch.tensor([]).to(device)
			for image_names, labels in train_loader1:
				inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
				features_batch = feature_model(inputs)

				features_batch = features_batch.view(features_batch.data.shape[0],512)

				gwr_labels_batch = labels // num_classes_in_task
		features = torch.cat((features, features_batch))
		gwr_labels = torch.cat((gwr_labels, gwr_labels_batch.float()))
		features = features.cpu()[::gwr_imgs_skip]
		gwr_labels = gwr_labels.cpu()[::gwr_imgs_skip]
		if(task_id == 0):
			gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs)	# GWR initiation
		else:
			gwr_model.train(features, gwr_labels, n_epochs = gwr_epochs, warm_start = True)	# continuing
	else:
		print('pretrained GWR given')

	num_nodes1 = gwr_model.get_num_nodes()		# currently existing no of nodes end of training
	print('number of nodes:', num_nodes1)
		
				
	return model, gwr_model


# def test_gwr(model, gwr_model, test_loader, test_loader_size, number_of_tasks,
# 					num_classes_in_task, KNN_k, individual_class,transform = None):
# 	feature_model = model
# 	model.eval()
# 	feature_model.eval()
# 	overall_corrects = 0
# 	gwr_corrects = 0
# 	gwr_category_by_category = torch.zeros(number_of_tasks)
# 	true_category_by_category = torch.zeros(number_of_tasks)
# 	nodes_per_tasks = gwr_model.nodes_per_task(number_of_tasks)
# 	with torch.set_grad_enabled(False):
# 		for image_names, labels in test_loader:
# 			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
# 			feature = feature_model(inputs)
# 			feature = feature.view(-1, 512).cpu()
# 			if not individual_class:
# 				gwr_label = (labels // num_classes_in_task).cpu()
# 			else:
# 				gwr_label = labels
# 			# print(image_names, gwr_label)
# 			fc_label = labels % num_classes_in_task
# 			pred_task = gwr_model.choose_task(feature, number_of_tasks,KNN_k)		# return predicted task ID, Note: line 304 and 308 doing same thing twice
# 			# print(pred_task)
# 			pred_task = torch.tensor(torch.from_numpy(pred_task), dtype = torch.int32).item()
# 			true_category_by_category[gwr_label] += 1
# 			if pred_task == gwr_label:
# 				gwr_corrects += 1
# 				gwr_category_by_category[gwr_label] += 1
# 	nodes_per_task1 = gwr_model.nodes_per_task(number_of_tasks)
# 	return gwr_corrects / test_loader_size * 100.0,gwr_category_by_category/true_category_by_category * 100, nodes_per_task1

def test_gwr(model, gwr_model, test_loader, test_loader_size, number_of_tasks,
					num_classes_in_task, KNN_k, individual_class,transform = None):
	number_of_tasks = 5
	feature_model = model
	model.eval()
	feature_model.eval()
	overall_corrects = 0
	gwr_corrects = 0
	gwr_category_by_category = torch.zeros(number_of_tasks)
	true_category_by_category = torch.zeros(number_of_tasks)
	nodes_per_tasks = gwr_model.nodes_per_task(10)
	with torch.set_grad_enabled(False):
		for image_names, labels in test_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			feature = feature_model(inputs)
			feature = feature.view(-1, 512).cpu()
			if not individual_class:
				gwr_label = (labels // 10).cpu()
			else:
				gwr_label = labels
			# print(image_names, gwr_label)
			fc_label = labels % num_classes_in_task
			pred_task = gwr_model.choose_task(feature, 10,KNN_k)		# return predicted task ID, Note: line 304 and 308 doing same thing twice
			# print(pred_task)
			pred_task = torch.tensor(torch.from_numpy(pred_task), dtype = torch.int32).item()
			pred_task = int(pred_task/2)
			# print(pred_task)
			true_category_by_category[gwr_label] += 1
			if pred_task == gwr_label:
				gwr_corrects += 1
				gwr_category_by_category[gwr_label] += 1
	nodes_per_task1 = gwr_model.nodes_per_task(10)
	return gwr_corrects / test_loader_size * 100.0,gwr_category_by_category/true_category_by_category * 100, nodes_per_task1

def test_gwr_fc(model, feature_model, gwr_model, criterion, test_loader, test_loader_size, number_of_tasks,
					num_classes_in_task, weight_dict, bias_dict,KNN_k,transform = None):
	# feature_model = nn.Sequential(*list(model.children())[:-1])
	model.eval()
	feature_model.eval()
	overall_corrects = 0
	gwr_corrects = 0
	predicted_class_by_class = torch.zeros(number_of_tasks*num_classes_in_task)
	true_class_by_class = torch.zeros(number_of_tasks*num_classes_in_task)
	gwr_category_by_category = torch.zeros(number_of_tasks)
	true_category_by_category = torch.zeros(number_of_tasks)
	nodes_per_tasks = gwr_model.nodes_per_task(number_of_tasks)
	with torch.set_grad_enabled(False):
		for image_names, labels in test_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			feature = feature_model(inputs)
			feature = feature.view(-1, 512).cpu()
			gwr_label = (labels // num_classes_in_task).cpu().item()
			fc_label = labels % num_classes_in_task
			pred_task = gwr_model.choose_task(feature, number_of_tasks,KNN_k)		# return predicted task ID, Note: line 304 and 308 doing same thing twice
			pred_task = torch.tensor(torch.from_numpy(pred_task), dtype = torch.int32).item()
			true_category_by_category[gwr_label] += 1
			if pred_task == gwr_label:
				gwr_corrects += 1
				gwr_category_by_category[gwr_label] += 1

			model.state_dict()['fc3.weight'].copy_(weight_dict[pred_task].to(device))  #vara
			model.state_dict()['fc3.bias'].copy_(bias_dict[pred_task].to(device))
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			true_class_by_class[labels] += 1
			if (pred_task == gwr_label) and (preds[0] == fc_label.data[0]):
				overall_corrects += 1
				predicted_class_by_class[pred_task*num_classes_in_task + preds[0]] += 1

	nodes_per_task1 = gwr_model.nodes_per_task(number_of_tasks)
	return gwr_corrects / test_loader_size * 100.0,gwr_category_by_category/true_category_by_category * 100,\
	overall_corrects / test_loader_size * 100.0,predicted_class_by_class/true_class_by_class * 100, nodes_per_task1

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

###  implemetation with pytorch - CPU version ###
class gwr_torch():

	def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
				 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, alpha_b = 1.05,
				 alpha_n = 1.05, h_0 = 1, sti_s = 1,
				 max_age = 100, max_size = 100,
				 random_state = None, device = "cpu" ):
		self.act_thr  = act_thr         #Actiavtion threshold
		self.fir_thr  = fir_thr         #Firing threshold
		self.eps_b    = eps_b           #epsilon b: Parmeter to control tuning the weights of BMU
		self.eps_n    = eps_n           #epsilon n: Parmeter to control tuning the weights of nodes adjacent to BMU
		self.tau_b    = tau_b           #Parameter to 
		self.tau_n    = tau_n
		self.alpha_b  = alpha_b
		self.alpha_n  = alpha_n
		self.h_0   = h_0
		self.sti_s   = sti_s
		self.max_age  = max_age
		self.max_size = max_size
		self.num_changes = 0
		self.device = device                #choosing device
		if random_state is not None:
			np.random.seed(random_state)

	def _initialize(self, X, Y):
		self.node_id = torch.tensor([]).int()
		self.pos = torch.tensor([]).float()
		self.fir = torch.tensor([]).float()
		self.n_best = torch.tensor([]).int()
		self.label = torch.tensor([]).float()
		self.best_act = torch.tensor([]).float()
		self.edges = {}

		draw = torch.randint(X.shape[0],(2,))

		#node : 1    
		self.node_id = torch.cat((self.node_id,torch.tensor([0]).int()))
		self.pos = torch.cat((self.pos,X[draw[0],:].view(1,-1)))
		self.label = torch.cat((self.label,Y[draw[0]].view(1)))
		self.fir = torch.cat((self.fir,torch.tensor([self.sti_s]).float()))
		self.n_best = torch.cat((self.n_best,torch.tensor([0]).int()))
		self.best_act = torch.cat((self.best_act,torch.tensor([1]).float()))
		self.edges[0] = {} 

		#node : 2    
		self.node_id = torch.cat((self.node_id,torch.tensor([1]).int()))
		self.pos = torch.cat((self.pos,X[draw[1],:].view(1,-1)))
		self.label = torch.cat((self.label,Y[draw[1]].view(1)))
		self.fir = torch.cat((self.fir,torch.tensor([self.sti_s]).float()))
		self.n_best = torch.cat((self.n_best,torch.tensor([0]).int()))
		self.best_act = torch.cat((self.best_act,torch.tensor([1]).float()))
		self.edges[1] = {} 

		self.last_node_idx = 1      #index of the last node created

		# print(self.node_id)
		# print(self.pos)

	def _get_best_matching(self, x):  
		dist = torch.cdist(x.view(1,-1), self.pos)
		sorted_dist = torch.argsort(dist.view(-1))
		# print('sorted_dist: ',sorted_dist)       
		b = self.node_id[sorted_dist[0]]
		s = self.node_id[sorted_dist[1]]
		id_b = sorted_dist[0]
		id_s = sorted_dist[1]

		self.n_best[sorted_dist[0]] += 1

		return id_b, id_s


	def _get_activation(self, x, id_b):
		pos_b = self.pos[id_b]
		dist = torch.dist(x, pos_b)
		# print(dist)
		act = torch.exp(-dist)
		return act


	def _make_link(self, id_b, id_s):
		b = int(self.node_id[id_b])
		s = int(self.node_id[id_s])
		if(s not in self.edges[b]):
			self.edges[b][s] = 0
			self.edges[s][b] = 0 

	def _add_node(self, x, y, id_b, id_s):
		r = self.last_node_idx + 1
		pos_r = 0.5 * (x + self.pos[id_b])
		dist = torch.dist(x, pos_r)
		act = torch.exp(-dist)

		#creating node
		self.node_id = torch.cat((self.node_id,torch.tensor([r]).int()))
		self.pos = torch.cat((self.pos,pos_r.view(1,-1)))
		self.label = torch.cat((self.label,y.view(1)))
		self.fir = torch.cat((self.fir,torch.tensor([self.sti_s]).float()))
		self.n_best = torch.cat((self.n_best,torch.tensor([0]).int()))
		self.best_act = torch.cat((self.best_act,torch.tensor([act]).float()))
		self.edges[r] = {}

		#removing edge between b,s
		b = int(self.node_id[id_b])
		s = int(self.node_id[id_s])
		if(s in self.edges[b]):
			del self.edges[b][s]
			del self.edges[s][b]

		#adding edges between (r,b) and (r,s) 
		self.edges[r][b] = 0
		self.edges[r][s] = 0
		self.edges[b][r] = 0
		self.edges[s][r] = 0

		self.last_node_idx = r
		return r

	def _update_network(self, x, id_b):
		dpos_b = self.eps_b * self.fir[id_b]*(x - self.pos[id_b])
		self.pos[id_b] = self.pos[id_b] + dpos_b

		b = int(self.node_id[id_b])
		for n in self.edges[b]:
			id_n = (self.node_id==n).nonzero(as_tuple = True)[0][0]
			# update the position of the neighbors
			dpos_n = self.eps_n * self.fir[id_n]*(x - self.pos[id_n])
			self.pos[id_n] = self.pos[id_n] + dpos_n  
			# increase the age of all edges connected to b    
			self.edges[b][n] += 1                   
			self.edges[n][b] += 1 

	def _update_firing(self, id_b):
		# self.fir[id_b] = self.h_0 - (self.sti_s/self.alpha_b)*\
		#     (1-torch.exp(-self.alpha_b*self.n_best[id_b]/self.tau_b))     
		dfir_b = (1/self.tau_b) * self.alpha_b*(1-self.fir[id_b]) - (1/self.tau_b)   
		self.fir[id_b] = torch.clamp(self.fir[id_b] + dfir_b,0.001,1)                   
		b = int(self.node_id[id_b])
		for n in list(self.edges[b]):
			id_n = (self.node_id==n).nonzero(as_tuple = True)[0][0]
			dfir_n = (1/self.tau_n) * self.alpha_n*(1-self.fir[id_n]) - (1/self.tau_n)
			self.fir[id_n] = torch.clamp(self.fir[id_n] + dfir_n,0.001,1)          
			# self.fir[id_n] = self.h_0 - (self.sti_s/self.alpha_n)*\
			#     (1-torch.exp(-self.alpha_n*self.n_best[id_n]/self.tau_n))

	def _remove_old_edges(self):
		for node_i in list(self.edges):
			for node_j in list(self.edges[node_i]):
				if(self.edges[node_i][node_j] > self.max_age):
					del self.edges[node_i][node_j]
					del self.edges[node_j][node_i]
		for node_i in list(self.edges):
			if(len(self.edges[node_i]) == 0):
				id_node_i = (self.node_id==node_i).nonzero(as_tuple = True)[0][0]
				self.node_id = torch.cat((self.node_id[:id_node_i],self.node_id[id_node_i+1:]))
				self.pos = torch.cat((self.pos[:id_node_i],self.pos[id_node_i+1:]))
				self.label = torch.cat((self.label[:id_node_i],self.label[id_node_i+1:]))
				self.fir = torch.cat((self.fir[:id_node_i],self.fir[id_node_i+1:]))
				self.n_best = torch.cat((self.n_best[:id_node_i],self.n_best[id_node_i+1:]))
				self.best_act = torch.cat((self.best_act[:id_node_i],self.best_act[id_node_i+1:]))
				del self.edges[node_i]

	def _check_stopping_criterion(self):
		# TODO: implement this
		pass

	def _training_step(self, x, y):
		id_b, id_s = self._get_best_matching(x)
		#print('best match ',b,s)            
		self._make_link(id_b, id_s)
		act = self._get_activation(x, id_b)
		fir = self.fir[id_b]

		if act < self.act_thr and fir < self.fir_thr \
			and len(self.node_id) < self.max_size:
			r = self._add_node(x, y, id_b, id_s)
			# logging.debug('GENERATE NODE %s', self.G.nodes[r])
		else:
			self._update_network(x, id_b)
			act = self._get_activation(x, id_b)
			if ((act > self.best_act[id_b]) and (int(self.label[id_b]) != int(y))):
				self.label[id_b] = y
				self.num_changes += 1

		self._update_firing(id_b)
		self._remove_old_edges()


	def train(self, X, Y, n_epochs=20, warm_start = False):
		if not warm_start:
			self._initialize(X,Y)
		print('training GWR..')
		# print(Y)
		for n in range(n_epochs):
			# print('gwr epoch: ',n)
			# logging.info('>>> Training epoch %s', str(n))
			for i in range(X.shape[0]):
				x = X[i]
				y = Y[i]
				self._training_step(x,y)
				self._check_stopping_criterion()
		print('num_changes: ', self.num_changes)
		self.num_changes = 0    

	def test(self, X, Y,num_tasks):                   
		num_correct = 0
		class_by_class = np.zeros(num_tasks)
		class_by_class_pred = np.zeros(num_tasks)
		for i in range(X.shape[0]):
			x = X[i]
			y = Y[i]
			class_by_class[int(y)] += 1
			id_b,id_s = self._get_best_matching(x)
			act = self._get_activation(x,id_b)
			y_pred = self.label[id_b]
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class
	  
	def _test_best_matching(self, x,k):          
		dist = torch.cdist(x.view(1,-1), self.pos)
		sorted_dist = torch.argsort(dist.view(-1))
		return sorted_dist[:k]
 
	def KNearest_test(self, X, Y,num_tasks,k):                   
		num_correct = 0
		class_by_class = np.zeros(num_tasks)
		class_by_class_pred = np.zeros(num_tasks)
		for i in range(X.shape[0]):
			x = X[i]
			y = Y[i]
			class_by_class[int(y)] += 1
			best_matches = self._test_best_matching(x,k)
			votes = np.zeros(num_tasks)
			for j in best_matches:
				votes[int(self.label[j])] += 1
			y_pred = np.argsort(votes)[-1]
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class

	def choose_task(self, X, num_tasks,k):
		pred_class = np.zeros(len(X))
		for i in range(X.shape[0]):
			x = X[i]
			# y = Y[i]
			best_matches = self._test_best_matching(x,k)
			votes = np.zeros(num_tasks)
			for j in best_matches:
				votes[int(self.label[j])] += 1
			y_pred = np.argsort(votes)[-1]
			# print(y_pred)
			pred_class[i] = y_pred
		return pred_class
		
	def nodes_per_task(self,num_tasks):
		tasks = np.zeros(num_tasks)
		for l in self.label:
			tasks[int(l)] += 1
		return tasks

	def get_num_nodes(self):
		return len(self.node_id)

#------------------------------------------------------------------------------------------------------------

###  gwr task wise learning implemetation with pytorch - CPU version ###
class gwr_task_torch():

	def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
				 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, alpha_b = 1.05,
				 alpha_n = 1.05, h_0 = 1, sti_s = 1,
				 max_age = 100, max_size = 100,
				 random_state = None, device = "cpu" ):
		self.act_thr  = act_thr         #Actiavtion threshold
		self.fir_thr  = fir_thr         #Firing threshold
		self.eps_b    = eps_b           #epsilon b: Parmeter to control tuning the weights of BMU
		self.eps_n    = eps_n           #epsilon n: Parmeter to control tuning the weights of nodes adjacent to BMU
		self.tau_b    = tau_b           #Parameter to 
		self.tau_n    = tau_n
		self.alpha_b  = alpha_b
		self.alpha_n  = alpha_n
		self.h_0   = h_0
		self.sti_s   = sti_s
		self.max_age  = max_age
		self.max_size = max_size
		self.num_changes = 0
		self.device = device                #choosing device
		if random_state is not None:
			np.random.seed(random_state)

		self.node_id = {}
		self.pos = {}	#torch.tensor([]).float()
		self.fir = {}	#torch.tensor([]).float()
		self.n_best = {}	#torch.tensor([]).int()
		self.label = {}	#torch.tensor([]).float()
		self.best_act = {}	#torch.tensor([]).float()
		self.edges = {}
		self.last_node_idx = {}


	def _initialize(self, X, Y, task_id):
		self.node_id[task_id] = torch.tensor([]).int()
		self.pos[task_id] = torch.tensor([]).float()
		self.fir[task_id] = torch.tensor([]).float()
		self.n_best[task_id] = torch.tensor([]).int()
		self.label[task_id] = torch.tensor([]).float()
		self.best_act[task_id] = torch.tensor([]).float()
		self.edges[task_id] = {}
		self.last_node_idx[task_id] = 0
		draw = torch.randint(X.shape[0],(2,))
		# print(X.shape,Y.shape)

		#node : 1    
		self.node_id[task_id] = torch.cat((self.node_id[task_id],torch.tensor([self.last_node_idx[task_id]]).int()))
		self.pos[task_id] = torch.cat((self.pos[task_id],X[draw[0],:].view(1,-1)))
		self.label[task_id] = torch.cat((self.label[task_id],Y[draw[0]].view(1)))
		self.fir[task_id] = torch.cat((self.fir[task_id],torch.tensor([self.sti_s]).float()))
		self.n_best[task_id] = torch.cat((self.n_best[task_id],torch.tensor([0]).int()))
		self.best_act[task_id] = torch.cat((self.best_act[task_id],torch.tensor([1]).float()))
		self.edges[task_id][0] = {} 
		self.last_node_idx[task_id] += 1

		#node : 2    
		self.node_id[task_id] = torch.cat((self.node_id[task_id],torch.tensor([self.last_node_idx[task_id]]).int()))
		self.pos[task_id] = torch.cat((self.pos[task_id],X[draw[1],:].view(1,-1)))
		self.label[task_id] = torch.cat((self.label[task_id],Y[draw[1]].view(1)))
		self.fir[task_id] = torch.cat((self.fir[task_id],torch.tensor([self.sti_s]).float()))
		self.n_best[task_id] = torch.cat((self.n_best[task_id],torch.tensor([0]).int()))
		self.best_act[task_id] = torch.cat((self.best_act[task_id],torch.tensor([1]).float()))
		self.edges[task_id][1] = {} 

		# self.last_node_idx += 1      #index of the last node created

		# print(self.node_id)
		# print(self.pos)

	def _get_best_matching(self, x,task_id):  
		dist = torch.cdist(x.view(1,-1), self.pos[task_id],p=1)     #p changed
		sorted_dist = torch.argsort(dist.view(-1))
		# print('sorted_dist: ',sorted_dist)       
		b = self.node_id[task_id][sorted_dist[0]]
		s = self.node_id[task_id][sorted_dist[1]]
		id_b = sorted_dist[0]
		id_s = sorted_dist[1]

		self.n_best[task_id][sorted_dist[0]] += 1

		return id_b, id_s


	def _get_activation(self, x, id_b,task_id):
		pos_b = self.pos[task_id][id_b]
		dist = torch.dist(x, pos_b,p=1)       #p changed
		# print(dist)
		act = torch.exp(-dist)
		return act


	def _make_link(self, id_b, id_s,task_id):
		b = int(self.node_id[task_id][id_b])
		s = int(self.node_id[task_id][id_s])
		if(s not in self.edges[task_id][b]):
			self.edges[task_id][b][s] = 0
			self.edges[task_id][s][b] = 0 

	def _add_node(self, x, y, id_b, id_s,task_id):
		r = self.last_node_idx[task_id] + 1
		pos_r = 0.5 * (x + self.pos[task_id][id_b])
		dist = torch.dist(x, pos_r)
		act = torch.exp(-dist)

		#creating node
		self.node_id[task_id] = torch.cat((self.node_id[task_id],torch.tensor([r]).int()))
		self.pos[task_id] = torch.cat((self.pos[task_id],pos_r.view(1,-1)))
		self.label[task_id] = torch.cat((self.label[task_id],y.view(1)))
		self.fir[task_id] = torch.cat((self.fir[task_id],torch.tensor([self.sti_s]).float()))
		self.n_best[task_id] = torch.cat((self.n_best[task_id],torch.tensor([0]).int()))
		self.best_act[task_id] = torch.cat((self.best_act[task_id],torch.tensor([act]).float()))
		self.edges[task_id][r] = {}

		#removing edge between b,s
		b = int(self.node_id[task_id][id_b])
		s = int(self.node_id[task_id][id_s])
		if(s in self.edges[task_id][b]):
			del self.edges[task_id][b][s]
			del self.edges[task_id][s][b]

		#adding edges between (r,b) and (r,s) 
		self.edges[task_id][r][b] = 0
		self.edges[task_id][r][s] = 0
		self.edges[task_id][b][r] = 0
		self.edges[task_id][s][r] = 0

		self.last_node_idx[task_id] = r
		return r

	def _update_network(self, x, id_b,task_id):
		dpos_b = self.eps_b * self.fir[task_id][id_b]*(x - self.pos[task_id][id_b])
		self.pos[task_id][id_b] = self.pos[task_id][id_b] + dpos_b

		b = int(self.node_id[task_id][id_b])
		for n in self.edges[task_id][b]:
			id_n = (self.node_id[task_id]==n).nonzero(as_tuple = True)[0][0]
			# update the position of the neighbors
			dpos_n = self.eps_n * self.fir[task_id][id_n]*(x - self.pos[task_id][id_n])
			self.pos[task_id][id_n] = self.pos[task_id][id_n] + dpos_n  
			# increase the age of all edges connected to b    
			self.edges[task_id][b][n] += 1                   
			self.edges[task_id][n][b] += 1 

	def _update_firing(self, id_b,task_id):
		# self.fir[id_b] = self.h_0 - (self.sti_s/self.alpha_b)*\
		#     (1-torch.exp(-self.alpha_b*self.n_best[id_b]/self.tau_b))     
		dfir_b = (1/self.tau_b) * self.alpha_b*(1-self.fir[task_id][id_b]) - (1/self.tau_b)   
		self.fir[task_id][id_b] = torch.clamp(self.fir[task_id][id_b] + dfir_b,0.001,1)                   
		b = int(self.node_id[task_id][id_b])
		for n in list(self.edges[task_id][b]):
			id_n = (self.node_id[task_id]==n).nonzero(as_tuple = True)[0][0]
			dfir_n = (1/self.tau_n) * self.alpha_n*(1-self.fir[task_id][id_n]) - (1/self.tau_n)
			self.fir[task_id][id_n] = torch.clamp(self.fir[task_id][id_n] + dfir_n,0.001,1)          
			# self.fir[id_n] = self.h_0 - (self.sti_s/self.alpha_n)*\
			#     (1-torch.exp(-self.alpha_n*self.n_best[id_n]/self.tau_n))

	def _remove_old_edges(self):
		for i,edges in self.edges.items():
			for node_i in list(edges):
				for node_j in list(edges[node_i]):
					if(edges[node_i][node_j] > self.max_age):
						del edges[node_i][node_j]
						del edges[node_j][node_i]
			for node_i in list(edges):
				if(len(edges[node_i]) == 0):
					id_node_i = (self.node_id[i]==node_i).nonzero(as_tuple = True)[0][0]
					self.node_id[i] = torch.cat((self.node_id[i][:id_node_i],self.node_id[i][id_node_i+1:]))
					self.pos[i] = torch.cat((self.pos[i][:id_node_i],self.pos[i][id_node_i+1:]))
					self.label[i] = torch.cat((self.label[i][:id_node_i],self.label[i][id_node_i+1:]))
					self.fir[i] = torch.cat((self.fir[i][:id_node_i],self.fir[i][id_node_i+1:]))
					self.n_best[i] = torch.cat((self.n_best[i][:id_node_i],self.n_best[i][id_node_i+1:]))
					self.best_act[i] = torch.cat((self.best_act[i][:id_node_i],self.best_act[i][id_node_i+1:]))
					del edges[node_i]

	def _check_stopping_criterion(self):
		# TODO: implement this
		pass

	def _training_step(self, x, y,task_id):
		id_b, id_s = self._get_best_matching(x,task_id)
		#print('best match ',b,s)            
		self._make_link(id_b, id_s,task_id)
		act = self._get_activation(x, id_b,task_id)
		fir = self.fir[task_id][id_b]

		if act < self.act_thr and fir < self.fir_thr \
			and len(self.node_id[task_id]) < self.max_size:
			r = self._add_node(x, y, id_b, id_s,task_id)
			# logging.debug('GENERATE NODE %s', self.G.nodes[r])
		else:
			self._update_network(x, id_b,task_id)
			# act = self._get_activation(x, id_b,task_id)
			# if ((act > self.best_act[task_id][id_b]) and (int(self.label[task_id][id_b]) != int(y))):
			# 	self.label[task_id][id_b] = y
			# 	self.num_changes += 1

		self._update_firing(id_b,task_id)
		self._remove_old_edges()


	def train(self, X, Y, task_id, n_epochs=20, new_task = False):
		if new_task:
			self._initialize(X,Y,task_id)
		print('training GWR..')
		act_thr = self.act_thr
		self.act_thr = np.exp(-4)
		# print(act_thr)
		feature_count = 0
		# print(Y)
		for n in range(n_epochs):
			# print('gwr epoch: ',n)
			# logging.info('>>> Training epoch %s', str(n))
			for i in range(X.shape[0]):
				x = X[i]
				y = Y[i]
				self._training_step(x,y,task_id)
				self._check_stopping_criterion()
				feature_count += 1
				if(feature_count>50):
					self.act_thr = act_thr
		# print('num_changes: ', self.num_changes)
		# self.num_changes = 0    

	def test(self, X, Y,num_tasks):                   
		num_correct = 0
		class_by_class = np.zeros(num_tasks)
		class_by_class_pred = np.zeros(num_tasks)
		for i in range(X.shape[0]):
			x = X[i]
			y = Y[i]
			class_by_class[int(y)] += 1
			id_b,id_s = self._get_best_matching(x)
			act = self._get_activation(x,id_b)
			y_pred = self.label[id_b]
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class
	  
	def _test_best_matching(self, x,k):
		Dist = torch.tensor([])
		for i,pos in self.pos.items():
			# print('pos: ' ,pos.shape)
			# print('x: ',x.view(1,-1).shape)
			dist = torch.cdist(x.view(1,-1), pos, p=1)  #p changed
			sorted_dist,sort_id = torch.sort(dist.view(-1))
			Dist = torch.cat((Dist,sorted_dist[:k]))
		sorted_Dist,sorted_ids = torch.sort(Dist.view(-1))

		channel_wise_thres = torch.zeros(k)
		for i,ids in enumerate(sorted_ids[:k]):
			channel_wise_thres[i] += torch.sum(torch.where(torch.exp(-torch.nan_to_num(torch.sqrt(x**2-pos[ids//k][ids%k]**2), nan=1e5)) > 0.5,1,0))
			# print(torch.exp(-torch.nan_to_num(torch.sqrt(x**2-pos[ids//k][ids%k]**2),nan=1e5)))		
		sorted_ids = sorted_ids // k
		return sorted_ids[:k], channel_wise_thres
 
	def KNearest_test(self, X, Y,num_tasks,k):                   
		num_correct = 0
		class_by_class = np.zeros(num_tasks)
		class_by_class_pred = np.zeros(num_tasks)
		for i in range(X.shape[0]):
			x = X[i]
			y = Y[i]
			class_by_class[int(y)] += 1
			best_matches = self._test_best_matching(x,k)
			# print(best_matches)
			votes = np.zeros(num_tasks)
			for j in best_matches:
				votes[j] += 1
			y_pred = np.argsort(votes)[-1]
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class

	# def choose_task(self, X, num_tasks,k):
	# 	# pred_class = np.zeros(len(X))
	# 	# print(X.shape)
	# 	for i in range(X.shape[0]):
	# 		x = X[i]
	# 		# y = Y[i]
	# 		best_matches, distances = self._test_best_matching(x,k)
	# 		votes = np.zeros(num_tasks)
	# 		for j in best_matches:
	# 			votes[j] += 1
	# 		task_occ = np.argsort(votes)
	# 		num_dist_smaples = votes[task_occ[-2]]
	# 		y_pred = {task_occ[-1]:0,task_occ[-2]:0}
	# 		for p,j in enumerate(best_matches):
	# 			if j in y_pred.keys():
	# 				y_pred[j] += distances[p] 
	# 		# print(y_pred)
	# 		# pred_class[i] = y_pred
	# 	return y_pred


	def choose_task(self, X, num_tasks,k):
		# pred_class = np.zeros(len(X))
		# print(X.shape)
		for i in range(X.shape[0]):
			x = X[i]
			# y = Y[i]
			best_matches, distances = self._test_best_matching(x,k)
			position_importance = torch.tensor([(1.1-i/10) for i in range(1,k+1)]) #(1.1-i/10) / 1/i**0.5
			votes = torch.zeros(num_tasks, dtype=torch.float32)
			dist_sum = torch.zeros(num_tasks, dtype=torch.float32)
			# for p,j in enumerate(best_matches):
				# dist_sum[j] += distances[p]
				# votes[j] += distances[p] #position_importance[p] #* torch.exp(-distances[p])#(1/distances[p]**0.75)
			task_occ = torch.argsort(distances)#(votes) #torch.nan_to_num(dist_sum/votes, nan=1e5),descending=True)
			distances = torch.nn.functional.softmax(distances,dim=0)#(votes/torch.sum(votes)) #torch.exp(-torch.nan_to_num(dist_sum/votes, nan=1e5))
			y_pred = {int(best_matches[task_occ[-1]]):distances[int(task_occ[-1])],int(best_matches[task_occ[-2]]):distances[int(task_occ[-2])]}  #{int(task_occ[-1]):votes[int(task_occ[-1])],int(task_occ[-2]):votes[int(task_occ[-2])]}
			# y_pred_list = list(y_pred.keys())
			# for p,j in enumerate(best_matches):
			# 	if j in y_pred_list:
			# 		y_pred_list.remove(j)
			# 		y_pred[int(j)] += distances[p] 
			# print(y_pred)
			# pred_class[i] = y_pred
		return y_pred
		
	def nodes_per_task(self,num_tasks):
		tasks = np.zeros(num_tasks)
		for n,n_id in self.node_id.items():
			tasks[n] = len(n_id)			
		return tasks

	def get_num_nodes(self):
		return len(self.node_id)

	def getpos(self):
		return self.pos

	def getlabels(self):
		return self.label






#------------------------------------------------------------------------------------------------------------

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


