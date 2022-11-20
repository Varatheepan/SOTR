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
import os
import copy
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def sof_max_update(model, criterion, optimizer, scheduler, train_loader,
					soft_dict, transform = None):
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
		new_distribution = F.softmax(outputs[i], dim = 0)#[:5]
		old_distribution = soft_dict[idx[i].item()]#[:5]
		kl_div_loss += klloss(new_distribution.log(), old_distribution)
	kl_div_loss /= len(idx)
	return kl_div_loss

def train_lwf(model, criterion, optimizer, scheduler, train_loader, task_id,
				soft_dict, lamda, num_epochs, transform = None):
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


