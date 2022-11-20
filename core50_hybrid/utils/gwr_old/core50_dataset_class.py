#################################################
	# Kanagarajah Sathursan
	# ksathursan1408@gmail.com
#################################################
from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse

#################################################
	# CORe50 Dataset Class
#################################################
class CORe50(Dataset):
	def __init__(self, train_csv_file, test_csv_file, images_dir, args):
		self.train_image_path_and_labels = np.array(pd.read_csv(train_csv_file))
		self.test_image_path_and_labels = np.array(pd.read_csv(test_csv_file))
		self.images_dir = images_dir
		self.args = args

	def get_train_set(self, idx = None):
		if idx:
			image_name = self.train_image_path_and_labels[idx][0]
			label = self.train_image_path_and_labels[idx][self.args.label_id]
			sample = [image_name, label - 1]
		else:
			sample = []
			for i in range(len(self.train_image_path_and_labels)):
				image_name = self.train_image_path_and_labels[i][0]
				label = self.train_image_path_and_labels[i][self.args.label_id]
				sample.append([image_name, label - 1])
		return sample

	def get_test_set(self, idx = None):
		if idx:
			image_name = self.test_image_path_and_labels[idx][0]
			label = self.test_image_path_and_labels[idx][self.args.label_id]
			sample = [image_name, label - 1]
		else:
			sample = []
			for i in range(len(self.test_image_path_and_labels)):
				image_name = self.test_image_path_and_labels[i][0]
				label = self.test_image_path_and_labels[i][self.args.label_id]
				sample.append([image_name, label - 1])
		return sample

	def get_train_by_task(self, mode, task_id):
		if mode == 'incremental_ni':
			instances = [1, 2, 4, 5, 6, 8, 9, 11]
			sample = []
			no_of_objects = 50 
			min_no_of_images_in_obj = 292
			rough_start_index = no_of_objects*min_no_of_images_in_obj*(task_id - 1)
			for i in range(rough_start_index , len(self.train_image_path_and_labels)):	# here rough_start_index is used to reduce the loop count
				if self.train_image_path_and_labels[i][1] == instances[task_id - 1]:
					image_name = self.train_image_path_and_labels[i][0]
					label = self.train_image_path_and_labels[i][self.args.label_id]
					sample.append([image_name, label - 1])
				elif self.train_image_path_and_labels[i][1] > instances[task_id - 1]:
					break

		elif mode == 'incremental_nc':
			sample = []
			label_sorted_list = self.train_image_path_and_labels[self.train_image_path_and_labels[:,self.args.label_id].argsort()]
			no_of_ses = 8
			min_no_of_images_in_obj = 292
			rough_start_index = no_of_ses*min_no_of_images_in_obj*self.args.shifter*(task_id - 1)
			for i in range(rough_start_index, len(label_sorted_list)): 
				if label_sorted_list[i][self.args.label_id] == task_id:
					image_name = label_sorted_list[i][0]
					label = label_sorted_list[i][self.args.label_id]
					sample.append([image_name, label - 1])
				elif label_sorted_list[i][self.args.label_id] == task_id + 1:
					break

		elif mode == 'incremental_nc_lwf':
			sample = []
			label_sorted_list = self.train_image_path_and_labels[self.train_image_path_and_labels[:,self.args.label_id].argsort()]
			no_of_ses = 8
			min_no_of_images_in_obj = 292
			rough_start_index = no_of_ses*min_no_of_images_in_obj*self.args.shifter*(task_id - 1)
			for i in range(rough_start_index, len(label_sorted_list)): 
				if label_sorted_list[i][self.args.label_id] == task_id:
					image_name = label_sorted_list[i][0]
					label = label_sorted_list[i][self.args.label_id]
					index = label_sorted_list[i][7]
					sample.append([image_name, label - 1, index])
				elif label_sorted_list[i][self.args.label_id] == task_id + 1:
					break

		elif mode == 'incremental_nic':
			sample = []
			min_no_of_images_in_obj = 292
			rough_start_index = min_no_of_images_in_obj*self.args.shifter*(task_id - 1)
			for i in range(rough_start_index , len(self.train_image_path_and_labels)):
				if self.train_image_path_and_labels[i][self.args.task_label_id] == task_id:
					image_name = self.train_image_path_and_labels[i][0]
					label = self.train_image_path_and_labels[i][self.args.label_id]
					sample.append([image_name, label - 1])
				elif self.train_image_path_and_labels[i][self.args.task_label_id] == task_id + 1:
					break
		return sample

	def get_test_by_task(self, mode, task_id):
		if mode == 'incremental_nc' or mode == 'incremental_nc_lwf':
			sample = []
			label_sorted_list = self.test_image_path_and_labels[self.test_image_path_and_labels[:,self.args.label_id].argsort()]
			no_of_ses = 3
			min_no_of_images_in_obj = 292
			rough_start_index = no_of_ses*min_no_of_images_in_obj*self.args.shifter*(task_id - 1)
			for i in range(rough_start_index, len(label_sorted_list)): 
				if label_sorted_list[i][self.args.label_id] == task_id:
					image_name = label_sorted_list[i][0]
					label = label_sorted_list[i][self.args.label_id]
					sample.append([image_name, label - 1])
				elif label_sorted_list[i][self.args.label_id] == task_id + 1:
					break
		return sample

