import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
'''
class MNIST_Net(nn.Module):
	def __init__(self):
		super(MNIST_Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc = nn.Linear(4*4*50, 10)
		
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*50)
		x = self.fc(x)
		return x
'''

################################################
	# Common
################################################

class MNIST_Net(nn.Module):
	def __init__(self, number_of_class):
		super(MNIST_Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc = nn.Linear(4*4*50, number_of_class)
		
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*50)
		x = self.fc(x)
		return x

'''
################################################
	# Latent comparison
################################################

class MNIST_Net(nn.Module):
	def __init__(self, number_of_class):
		super(MNIST_Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, 5, 1)
		self.conv2 = nn.Conv2d(10, 20, 5, 1)
		self.conv3 = nn.Conv2d(20, 50, 3, 1)
		self.fc = nn.Linear(2*2*50, number_of_class)
		
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv3(x))
		x = x.view(-1, 2*2*50)
		x = self.fc(x)
		return x
'''
'''
################################################
	# Linear Network
################################################

class MNIST_Net(nn.Module):
	def __init__(self, number_of_class):
		super(MNIST_Net, self).__init__()
		self.fc1 = nn.Linear(784, 400)
		self.fc2 = nn.Linear(400, 400)
		self.fc3 = nn.Linear(400, number_of_class)


	def forward(self, x):
		x = x.view(-1, 784)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
'''