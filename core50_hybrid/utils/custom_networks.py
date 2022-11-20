from __future__ import print_function, division
import torch 
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import time
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import sys
sys.path.insert(0, '../../utils')
from util_funcs import *
from custom_datasets import *
from gwr import GWR
from associative_memory import associative_memory
import math 
from resnet_cifar import *

class lenet_gwr(nn.Module):
    def __init__(self, args):
        super(lenet_gwr, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.fc2 = nn.Linear(256, 10)

        # gwr params
        # self.net.to(args.device)
        self.num_labels = args.num_labels
        self.gwr_enable_quantization = args.gwr_enable_quantization
        self.gwr_quantization_thresh = args.gwr_quantization_thresh
        self.gwr =  GWR(activity_thr=args.gwr_activity_thresh, firing_counter=args.gwr_firing_counter, epsn=args.gwr_epsn, epsb=args.gwr_epsb, 
                        enable_quantization=self.gwr_enable_quantization, quantization_thresh=self.gwr_quantization_thresh, max_edge_age=args.gwr_max_edge_age)
        self.associative_mem = associative_memory(1, 0.1)
        self.gwr.register_on_train_sample_callback(self.train_assoc_mem)
        self.gwr.register_on_test_sample_callback(self.test_assoc_mem)
        self.train_test_mode = 0
        self.class_idx = 0
        self.gwr_num_epochs = args.gwr_num_epochs
        self.gwr_cnn_fvec_layer = args.gwr_cnn_fvec_layer
        self.is_trained = False
    
    def set_run_mode(self,mode, class_idx):
        self.train_test_mode = mode
        self.class_idx       = class_idx

    def forward(self, x, y=None):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4*4*50)
        # x = F.relu(self.fc1(x))
        #bypass final fully connected layers
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        batch_size = x.size(0)
        feature_vec_batch = x.clone()
        if(y is not None) : 
            y = y.cpu().numpy()
        correct_count = 0
        prediction_samples = []
        if(self.train_test_mode == TRAIN_MODE) : 
            # if(self.class_idx > 1): 
            # print("y 0 : ", y[0])
            feature_vec_batch = feature_vec_batch.cpu().numpy()
            if(self.is_trained == True) : 
                # gwr.fit(current_class_train_data, y=current_class_train_label, normalize=False, warm_start=True, iters=5)
                self.gwr.fit(feature_vec_batch, y=y, normalize=False, warm_start=True, iters=self.gwr_num_epochs , verbose=True) #self.gwr_num_epochs)
            else :
                self.gwr.fit(feature_vec_batch, y=y, normalize=False, iters=self.gwr_num_epochs, verbose=True) #self.gwr_num_epochs)
                self.is_trained = True
        elif(self.train_test_mode == TEST_MODE) : 
            feature_vec_batch = feature_vec_batch.cpu().numpy()
            prediction_samples = self.gwr.predict(feature_vec_batch, y, labeldim=self.num_labels)
        return correct_count, prediction_samples, feature_vec_batch

    #################################################
    # custom functions
    #################################################
    def train_assoc_mem(self, neuron_id, label) : 
        self.associative_mem.train_mem(neuron_id, label)
        
    def test_assoc_mem(self, neuron_id, label) : 
        # global correct_count
        correct_count = 0
        pred_label = self.associative_mem.get_label(neuron_id)
        ground_truth_label = label
        return pred_label
    
'''*************************************************************************'''
class cnn_gwr(nn.Module):

    def __init__(self, args):
        super(cnn_gwr, self).__init__()
        # if(args.model is "resnet18")  :
        self.net = torchvision.models.resnet18(pretrained=True)
        #self.net = torchvision.models.resnet34(pretrained=True)
        self.net.to(args.device)
        self.num_labels = args.num_labels
        # print("net.........", self.net)
        # else if(args.model is "mnist")  :
        #     self.net = torchvision.models.resnet18(pretrained=True)
        # gwr_activity_thr = 0.6
        
        self.gwr_enable_quantization = args.gwr_enable_quantization
        self.gwr_quantization_thresh = args.gwr_quantization_thresh
        self.gwr =  GWR(activity_thr=args.gwr_activity_thresh, firing_counter=args.gwr_firing_counter, epsn=args.gwr_epsn, epsb=args.gwr_epsb, 
                        enable_quantization=self.gwr_enable_quantization, quantization_thresh=self.gwr_quantization_thresh, max_edge_age=args.gwr_max_edge_age)
        # self.gwr = GWR(activity_thr=args.gwr_activity_thresh)
        self.associative_mem = associative_memory(1, 0.1)
        self.gwr.register_on_train_sample_callback(self.train_assoc_mem)
        self.gwr.register_on_test_sample_callback(self.test_assoc_mem)
        self.train_test_mode = 0
        self.class_idx = 0
        self.gwr_num_epochs = args.gwr_num_epochs
        self.gwr_cnn_fvec_layer = args.gwr_cnn_fvec_layer
        self.is_trained = False

    def set_run_mode(self,mode, class_idx):
        self.train_test_mode = mode
        self.class_idx       = class_idx

    def forward(self, x, y=None, confusion_matrix=None):
        batch_size = x.size(0)
        layer = self.net._modules.get(self.gwr_cnn_fvec_layer)

        feature_vec_batch = torch.zeros(batch_size, 512, 1, 1)
        def copy_data(m, i, o):
            if(o.data.size() != feature_vec_batch.size()) : 
                print("size mismatch, O : ",o.data.size(),  "FVec :  ", feature_vec_batch.size())      
            else : 
                feature_vec_batch.copy_(o.data)
        h = layer.register_forward_hook(copy_data)

        out = self.net(x)
        h.remove()

        feature_vec_batch = feature_vec_batch.view(feature_vec_batch.size(0), -1)
        if(y is not None) : 
            y = y.cpu().numpy()
        correct_count = 0
        prediction_samples = []
        if(self.train_test_mode == TRAIN_MODE) : 
            # if(self.class_idx > 1): 
            print("y 0 : ", y[0])
            feature_vec_batch = feature_vec_batch.cpu().numpy()
            if(self.is_trained == True) : 
                # gwr.fit(current_class_train_data, y=current_class_train_label, normalize=False, warm_start=True, iters=5)
                self.gwr.fit(feature_vec_batch, y=y, normalize=False, warm_start=True, iters=self.gwr_num_epochs , verbose=True) #self.gwr_num_epochs)
            else :
                self.gwr.fit(feature_vec_batch, y=y, normalize=False, iters=self.gwr_num_epochs, verbose=True) #self.gwr_num_epochs)
                self.is_trained = True
        elif(self.train_test_mode == TEST_MODE) : 
            feature_vec_batch = feature_vec_batch.cpu().numpy()
            prediction_samples = self.gwr.predict(feature_vec_batch, y, labeldim=self.num_labels)
        return correct_count, prediction_samples, feature_vec_batch

    #################################################
    # custom functions
    #################################################
    def train_assoc_mem(self, neuron_id, label) : 
    #     print("Neuron ID : %d, label : %d" %(neuron_id, label))
        self.associative_mem.train_mem(neuron_id, label)
        
    def test_assoc_mem(self, neuron_id, label) : 
        # global correct_count
        correct_count = 0
        pred_label = self.associative_mem.get_label(neuron_id)
        # print("Neuron ID : %d, label : %d, Pred Label : %d" %(neuron_id, label, pred_label))
        ground_truth_label          = label
        # ground_truth_category_label = int(math.floor(ground_truth_label/5))
        # pred_category_label         = int(math.floor(pred_label/5)) 
        # print("pred category : %d pred object : %d" %(pred_category_label, pred_label))
        # return pred_category_label, pred_label
        return pred_label
    

# class ResNet_cifar_gwr(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10, args=None):
#         super(ResNet_cifar_gwr, self).__init__()
#         self.in_planes = 64

#         # layer size configuration
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, 10) #test #num_classes)

#         self.num_labels = args.num_labels
#         self.gwr_enable_quantization = args.gwr_enable_quantization
#         self.gwr_quantization_thresh = args.gwr_quantization_thresh
#         self.gwr =  GWR(activity_thr=args.gwr_activity_thresh, firing_counter=args.gwr_firing_counter, epsn=args.gwr_epsn, epsb=args.gwr_epsb, 
#                         enable_quantization=self.gwr_enable_quantization, quantization_thresh=self.gwr_quantization_thresh, max_edge_age=args.gwr_max_edge_age)
#         self.associative_mem = associative_memory(1, 0.1)
#         self.gwr.register_on_train_sample_callback(self.train_assoc_mem)
#         self.gwr.register_on_test_sample_callback(self.test_assoc_mem)
#         self.train_test_mode = 0
#         self.class_idx = 0
#         self.gwr_num_epochs = args.gwr_num_epochs
#         self.gwr_cnn_fvec_layer = args.gwr_cnn_fvec_layer
#         self.is_trained = False

#     def set_run_mode(self,mode, class_idx):
#         self.train_test_mode = mode
#         self.class_idx       = class_idx

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x, y=None):
#         batch_size = x.size(0)
        
#         print("x : ", x.shape)
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         feature_vec_batch = out.clone()
#         out = self.linear(out)

#         # feature_vec_batch = feature_vec_batch.view(feature_vec_batch.size(0), -1)
#         # feature_vec_batch = feature_vec_batch
#         if(y is not None) : 
#             y = y.cpu().numpy()
#         correct_count = 0
#         prediction_samples = []
#         if(self.train_test_mode == TRAIN_MODE) : 
#             # print("y 0 : ", y[0])
#             feature_vec_batch = feature_vec_batch.cpu().numpy()
#             if(self.is_trained == True) : 
#                 self.gwr.fit(feature_vec_batch, y=y, normalize=False, warm_start=True, iters=self.gwr_num_epochs , verbose=True) #self.gwr_num_epochs)
#             else :
#                 self.gwr.fit(feature_vec_batch, y=y, normalize=False, iters=self.gwr_num_epochs, verbose=True) #self.gwr_num_epochs)
#                 self.is_trained = True
#         elif(self.train_test_mode == TEST_MODE) : 
#             feature_vec_batch = feature_vec_batch.cpu().numpy()
#             prediction_samples = self.gwr.predict(feature_vec_batch, y, labeldim=self.num_labels)
       
#         return correct_count, prediction_samples, feature_vec_batch    
#         # return out

#     def train_assoc_mem(self, neuron_id, label) : 
#         self.associative_mem.train_mem(neuron_id, label)
    
#     def test_assoc_mem(self, neuron_id, label) : 
#         correct_count = 0
#         pred_label = self.associative_mem.get_label(neuron_id)
#         ground_truth_label          = label
#         return pred_label

class cifar10_cnn_gwr(nn.Module):
    def __init__(self, args):
        super(cifar10_cnn_gwr, self).__init__()
        # self.net = torchvision.models.resnet18(pretrained=True)
        self.net = ResNet18()
        self.net.to(args.device)
        self.num_labels = args.num_labels
        
        self.gwr_enable_quantization = args.gwr_enable_quantization
        self.gwr_quantization_thresh = args.gwr_quantization_thresh
        self.gwr =  GWR(activity_thr=args.gwr_activity_thresh, firing_counter=args.gwr_firing_counter, epsn=args.gwr_epsn, epsb=args.gwr_epsb, 
                        enable_quantization=self.gwr_enable_quantization, quantization_thresh=self.gwr_quantization_thresh, max_edge_age=args.gwr_max_edge_age)
        self.associative_mem = associative_memory(1, 0.1)
        self.gwr.register_on_train_sample_callback(self.train_assoc_mem)
        self.gwr.register_on_test_sample_callback(self.test_assoc_mem)
        self.train_test_mode = 0
        self.class_idx = 0
        self.gwr_num_epochs = args.gwr_num_epochs
        self.gwr_cnn_fvec_layer = args.gwr_cnn_fvec_layer
        self.is_trained = False

    def set_run_mode(self,mode, class_idx):
        self.train_test_mode = mode
        self.class_idx       = class_idx

    def forward(self, x, y=None, confusion_matrix=None):
        batch_size = x.size(0)
        layer = self.net._modules.get(self.gwr_cnn_fvec_layer)

        feature_vec_batch = torch.zeros(batch_size, 512, 1, 1)
        # def copy_data(m, i, o):
        #     if(o.data.size() != feature_vec_batch.size()) : 
        #         print("size mismatch, O : ",o.data.size(),  "FVec :  ", feature_vec_batch.size())      
        #     else : 
        #         feature_vec_batch.copy_(o.data)
        # h = layer.register_forward_hook(copy_data)

        out = self.net(x)
        # feature_vec_batch = out.cpu().numpy()
        # print("out shape : ", out.size())
        # raw_input()
        # h.remove()
        feature_vec_batch = out
        # print(feature_vec_batch)
        # raw_input()
        # print(feature_vec_batch.size())
        feature_vec_batch = feature_vec_batch.view(feature_vec_batch.size(0), -1)
        if(y is not None) : 
            y = y.cpu().numpy()
        correct_count = 0
        prediction_samples = []
        if(self.train_test_mode == TRAIN_MODE) : 
            print("y 0 : ", y[0])
            feature_vec_batch = feature_vec_batch.cpu().numpy()
            if(self.is_trained == True) : 
                # gwr.fit(current_class_train_data, y=current_class_train_label, normalize=False, warm_start=True, iters=5)
                self.gwr.fit(feature_vec_batch, y=y, normalize=False, warm_start=True, iters=self.gwr_num_epochs , verbose=True) #self.gwr_num_epochs)
            else :
                self.gwr.fit(feature_vec_batch, y=y, normalize=False, iters=self.gwr_num_epochs, verbose=True) #self.gwr_num_epochs)
                self.is_trained = True
        elif(self.train_test_mode == TEST_MODE) : 
            feature_vec_batch = feature_vec_batch.cpu().numpy()
            # correct_count = self.gwr.predict(feature_vec_batch, y, labeldim=self.num_labels)
            prediction_samples = self.gwr.predict(feature_vec_batch, y, labeldim=self.num_labels)
        return correct_count, prediction_samples, feature_vec_batch

    #################################################
    # custom functions
    #################################################
    def train_assoc_mem(self, neuron_id, label) : 
        # print("train assoc mem : neuron id :", neuron_id, " label : ",  label)
        self.associative_mem.train_mem(neuron_id, label)
    
    def test_assoc_mem(self, neuron_id, label) : 
        correct_count = 0
        pred_label = self.associative_mem.get_label(neuron_id)
        ground_truth_label          = label
        return pred_label


def resnet18_cifar_gwr(args) : 
    return cifar10_cnn_gwr(args)
