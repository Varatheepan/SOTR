#################################################
	# Kanagarajah Sathursan
	# ksathursan1408@gmail.com
#################################################
from __future__ import print_function, division
import torch
import os

class_by_class_dir = '../datasets/MNIST/class_by_class'
# class_by_class_dir = '../datasets/MNIST/class_by_class_with_idx'
cumulative_dir = '../datasets/MNIST/cumulative/'
train_dir = '../datasets/MNIST/processed/training.pt'
test_dir = '../datasets/MNIST/processed/test.pt'

train_set = torch.load(train_dir)
test_set = torch.load(test_dir)

#################################################
	# Cumulative
#################################################
train_cumulative = []
test_cumulative = []
for idx in range(len(train_set[1])):
	train_cumulative.append([torch.FloatTensor(train_set[0][idx].unsqueeze(dim=0).tolist()), train_set[1][idx]])		
torch.save(train_cumulative, cumulative_dir + 'train.pt')	

for idx in range(len(test_set[1])):
	test_cumulative.append([torch.FloatTensor(test_set[0][idx].unsqueeze(dim=0).tolist()), test_set[1][idx]])		
torch.save(test_cumulative, cumulative_dir + 'test.pt')	

#################################################
	# Incremental
#################################################
for class_id in range(10):
	train_class = []
	for idx in range(len(train_set[1])):
		if train_set[1][idx] == class_id:
			train_class.append([torch.FloatTensor(train_set[0][idx].unsqueeze(dim=0).tolist()), train_set[1][idx]])
			# train_class.append([torch.FloatTensor(train_set[0][idx].unsqueeze(dim=0).tolist()), train_set[1][idx], idx])

	torch.save(train_class, os.path.join(class_by_class_dir, 'train', 'class_' + str(class_id) + '.pt'))	

	test_class = []
	for idx in range(len(test_set[1])):
		if test_set[1][idx] == class_id:
			test_class.append([torch.FloatTensor(test_set[0][idx].unsqueeze(dim=0).tolist()), test_set[1][idx]])		
	torch.save(test_class, os.path.join(class_by_class_dir, 'test', 'class_' + str(class_id) + '.pt'))	

