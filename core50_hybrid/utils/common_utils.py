#################################################
	# Kanagarajah Sathursan
	# ksathursan1408@gmail.com
#################################################
from __future__ import print_function, division
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

#################################################
	# Accuracy of each classes
#################################################
def class_acc(number_of_correct_by_class, number_by_class):
	
	no_of_classes = len(number_by_class)
	class_accuracy = np.zeros((1, no_of_classes), dtype=int)
	for i in range(no_of_classes):
		if number_by_class[i] == 0:
			cl_acc = 0
		else:
			cl_acc = (number_of_correct_by_class[i] / number_by_class[i]) * 100
		class_accuracy[0][i] = cl_acc
	return class_accuracy

#################################################
	# Confusion matrix
#################################################
def plot_confusion_matrix(confusion_matrix, file_name, normalize =  False, title = None, cmap = plt.cm.Blues):
	classes = []
	for i in range(confusion_matrix.shape[0]):
		classes.append(i)
		
	if normalize:
		confusion_matrix = confusion_matrix.astype(float)
		for i in range(confusion_matrix.shape[0]):
			confusion_matrix[i] = confusion_matrix[i]/np.sum(confusion_matrix[i], dtype = float) if (sum(confusion_matrix[i])!= 0) else 0 

	fig, ax = plt.subplots()
	im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(confusion_matrix.shape[1]), yticks=np.arange(confusion_matrix.shape[0]),
		xticklabels=classes, yticklabels=classes,
		title=title,
		ylabel='True label',
		xlabel='Predicted label')

	fmt = '.2f' #if normalize else 'd'
	thresh = confusion_matrix.max() / 2.
	for i in range(confusion_matrix.shape[0]):
		for j in range(confusion_matrix.shape[1]):
			ax.text(j, i, format(confusion_matrix[i, j], fmt), ha="center", va="center", 
				color="white" if confusion_matrix[i, j] > thresh else "black")


	fig.tight_layout()
	plt.savefig(file_name)
	# plt.show()
	return ax