from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import os 
from PIL import Image
import torch
import random
import numpy as np
import pandas as pd
from skimage import io, transform

TRAIN_MODE = 0
TEST_MODE = 1
FEATURE_EXTRACT_MODE = 2

CUMULATIVE_MODE = 1
INCREMENTAL_MODE = 2
PLOT_MODE = 3

def default_loader(path):
    return Image.open(path).convert('RGB')

class CaltechDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader, 
                train_test_mode=TRAIN_MODE, num_test_img=10, num_train_img=10):
        # fh = open(txt, 'r')
        # imgs = []
        # for line in fh:
        #     line = line.rstrip()
        #     line = line.strip('\n')
        #     line = line.rstrip()
        #     words = line.split()
        #     imgs.append((words[0],int(words[1])))
        # print(imgs)
        self.imgs_dict = {}
        self.imgs = []
        self.data_load_mode = 0
        self.train_test_mode = train_test_mode
        self.class_idx = 0
        self.class_count = 0

        print(train_test_mode)
        class_idx = 0
        for class_dir in os.listdir(txt) :
            class_dir_path = txt + '/' + class_dir
            class_imgs = []
            img_count  = 0
            for file in os.listdir(class_dir_path) : 
                if(train_test_mode == TRAIN_MODE) :
                    if(img_count < num_test_img) :
                        img_count += 1
                        continue
                    elif(img_count == num_test_img + num_train_img) : 
                        break
                    else :
                        img_count += 1
                elif(train_test_mode == TEST_MODE) :
                    if(img_count==num_test_img) : 
                        break
                    else :
                        img_count += 1
                img_path = class_dir_path + '/' + file
                class_imgs.append(img_path)
                self.imgs.append( (img_path, class_idx) )
            print(class_idx, len(class_imgs))
            self.imgs_dict[class_idx] = class_imgs
            class_idx += 1
        self.class_count = class_idx
        # for img in self.imgs_dict[1] :
        #     print(img)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_data_mode(self, train_mode, class_idx=0):
        self.data_load_mode = train_mode
        self.class_idx = class_idx

    def set_data_mode_train(self):
        self.train_test_mode = TRAIN_MODE

    def set_data_mode_test(self):
        self.train_test_mode = TEST_MODE

    def get_class_count(self) : 
        return self.class_count

    # def __getitem__(self, index):
    #     fn, label = self.imgs[index]
    #     img = self.loader(fn)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img,label
    def __getitem__(self, index):
        if(self.data_load_mode == CUMULATIVE_MODE) : 
            fn, label = self.imgs[index]
        else :
            img = (self.imgs_dict[self.class_idx])[index]
            fn, label = img, self.class_idx
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        
        if(self.data_load_mode == INCREMENTAL_MODE) : 
            if(self.class_idx != label) : 
                print("ERROR in dataloader")
        return img,label

    def __len__(self):
        if(self.data_load_mode == CUMULATIVE_MODE) : 
            return len(self.imgs)
        else :
            # print(len(self.imgs_dict[self.class_idx]))
            return len(self.imgs_dict[self.class_idx])


class COIL100Dataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader, 
                train_test_mode=TRAIN_MODE, num_test_img=32, num_train_img=40):
        self.imgs_dict = {}
        self.imgs = []
        self.data_load_mode = 0
        self.train_test_mode = train_test_mode
        self.class_idx = 0
        self.class_count = 0
        self.num_img_per_class = 72 #360 degrees covered in 5 degree steps
        self.root_dir = txt
        file_list = os.listdir(txt)
        file_list = sorted(file_list)
        img_count = 0
        class_idx = 0
        class_imgs = []
        for file in file_list: #traverse the directory
            if(file.endswith('.png')) : #filter images
                class_imgs.append( (file, class_idx) )
                if(img_count == self.num_img_per_class-1) :
                    if(train_test_mode == TRAIN_MODE) :
                        self.imgs_dict[class_idx] = class_imgs[0:num_train_img]
                        self.imgs.extend(class_imgs[0:num_train_img])
                    elif(train_test_mode == TEST_MODE) :
                        self.imgs_dict[class_idx] = class_imgs[num_train_img:(num_test_img + num_train_img + 1)]
                        self.imgs.extend(class_imgs[num_train_img:(num_test_img + num_train_img + 1)])
                    img_count = 0
                    class_idx += 1
                    class_imgs = []
                else :
                    img_count += 1

        print(self.imgs[0:10])
                    
        # print(len(self.imgs_dict[0]))
        # print(len(self.imgs_dict[1]))
        # print(self.imgs[0:10])
        self.class_count = class_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_data_mode(self, train_mode, class_idx=0):
        self.data_load_mode = train_mode
        self.class_idx = class_idx

    def set_data_mode_train(self):
        self.train_test_mode = TRAIN_MODE

    def set_data_mode_test(self):
        self.train_test_mode = TEST_MODE

    def get_class_count(self) :
        print("Class Count : ", self.class_count) 
        return self.class_count

    # def __getitem__(self, index):
    #     fn, label = self.imgs[index]
    #     img = self.loader(fn)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img,label
    def __getitem__(self, index):
        if(self.data_load_mode == CUMULATIVE_MODE) : 
            fn, label = self.imgs[index]

            # print("cumulative get item")
        else :
            fn, label = (self.imgs_dict[self.class_idx])[index]
        fn = self.root_dir + '/' + fn
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        
        if(self.data_load_mode == INCREMENTAL_MODE) : 
            if(self.class_idx != label) : 
                print("ERROR in dataloader")
        return img,label

    def __len__(self):
        if(self.data_load_mode == CUMULATIVE_MODE) : 
            return len(self.imgs)
        else :
            # print(len(self.imgs_dict[self.class_idx]))
            return len(self.imgs_dict[self.class_idx])

def generate_mnist_permutation(seed) : 
    np.random.seed(seed)
    h = w = 28
    perm_inds = list(range(h*w))
    np.random.shuffle(perm_inds)
    return perm_inds

class MNISTDataset(Dataset):
    def __init__(self, txt=None, transform=None, target_transform=None, loader=default_loader, 
                train_test_mode=TRAIN_MODE, num_test_img=10, num_train_img=10, enable_permutations=False, permutation_idx=None):
        # fh = open(txt, 'r')
        # imgs = []
        # for line in fh:
        #     line = line.rstrip()
        #     line = line.strip('\n')
        #     line = line.rstrip()
        #     words = line.split()
        #     imgs.append((words[0],int(words[1])))
        # print(imgs)
        # self.imgs_dict = {}
        # self.imgs = []
        self.data_load_mode = 0
        self.train_test_mode = train_test_mode
        self.class_idx = 0
        self.class_count = 0
        # self.processed_folder = './data/processed'
        self.processed_folder = './data/MNIST/processed'
        self.class_data = []
        self.class_labels = []
        self.class_count = 10
        if(self.train_test_mode == TRAIN_MODE):
            self.dataset = datasets.MNIST('./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
            self.data, self.labels = torch.load(os.path.join(self.processed_folder, self.dataset.training_file))

            if(enable_permutations == True) : 
                self.data = torch.stack([ (img.float().view(-1)[permutation_idx]).view(28,28)
                                           for img in self.data])

            # self.data = self.data[0:5000]
            # self.labels = self.labels[0:5000]
            
            # for i in range(10) : 
            #     print("Digit %d frequence : %d" %(i, sum(self.labels==i)))
            #downsize the dataset
            # print(self.data.size())
            # print(self.labels.size())
            temp_data = []
            temp_labels = []
            for i in range(self.class_count):
                class_i_data = self.data[self.labels==i]
                class_i_labels = self.labels[self.labels==i]
                class_i_data = class_i_data[0:500]
                class_i_labels = class_i_labels[0:500]

                for j in range(500):
                    temp_data.append(class_i_data[j])
                    temp_labels.append(class_i_labels[j])
            temp_data_arry = list(zip(temp_data, temp_labels))
            # print(temp_labels[499], temp_labels[500], temp_labels[1000])
            # print(temp_data[499], temp_data[500], temp_data[1000])
            random.shuffle(temp_data_arry)
            shuffled_data, shuffled_labels = zip(*temp_data_arry)   
            shuffled_data = list(shuffled_data)     
            shuffled_labels = list(shuffled_labels)      
            
            self.data = []
            self.labels = []
            for data in shuffled_data : 
                data = data.unsqueeze(0)
                if(len(self.data) == 0) : 
                    self.data = data
                else : 
                    self.data = torch.cat((self.data, data), 0)
                # print(self.data.size())
                # print(data.size()) 
            self.labels = torch.tensor(shuffled_labels)
        else :
            self.dataset = datasets.MNIST('./data', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
            self.data, self.labels = torch.load(os.path.join(self.processed_folder, self.dataset.test_file))
            if(enable_permutations == True) : 
                self.data = torch.stack([ (img.float().view(-1)[permutation_idx]).view(28,28)
                                           for img in self.data])
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        print("generating MNIST data succesful...", self.data.size(), self.labels.size())


    def set_data_mode(self, train_mode, class_idx=0):
        self.data_load_mode = train_mode
        self.class_idx = class_idx
        self.class_data = self.data[self.labels == class_idx]
        self.class_labels = self.labels[self.labels == class_idx]

    def set_data_mode_train(self):
        self.train_test_mode = TRAIN_MODE

    def set_data_mode_test(self):
        self.train_test_mode = TEST_MODE

    def get_class_count(self) : 
        return self.class_count
    
    # def __getitem__(self, index):
    #     fn, label = self.imgs[index]
    #     img = self.loader(fn)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img,label
    def __getitem__(self, index):
        if(self.data_load_mode == CUMULATIVE_MODE) : 
            # fn, label = self.imgs[index]
            fn = self.data[index]
            label = self.labels[index]
        else :
            fn = self.class_data[index]
            label = self.class_labels[index]
        
        # img = self.loader(fn)
        # print("get_item : img shape : ", fn.size(), " label shape : ", label.size())
        img = Image.fromarray(fn.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        
        if(self.data_load_mode == INCREMENTAL_MODE) : 
            if(self.class_idx != label) : 
                print("ERROR in dataloader")
        return img,label

    def __len__(self):
        if(self.data_load_mode == CUMULATIVE_MODE) : 
            return len(self.labels)
        else :
            # print(len(self.imgs_dict[self.class_idx]))
            return len(self.class_labels)

##############################################
# CIFAR-100
##############################################

class CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self, txt=None, transform=None, target_transform=None, loader=default_loader, 
                train_test_mode=TRAIN_MODE, num_test_img=10, num_train_img=10):
        
        if(train_test_mode == TRAIN_MODE) : 
            super(CIFAR100Dataset, self).__init__('data', train=True, download=True,    
                                            transform=transform)
        elif(train_test_mode == TEST_MODE) : 
            super(CIFAR100Dataset, self).__init__('data', train=False, download=True,
                                            transform=transform)
    # def __getitem__(self, index):
    #     img, label = super.__getitem__(index)
    #     print("img : ", img, label)
    #     return img, label

    # def __len__(self):
        
    #     len = super.__len__()
    #     print("len : ", len)
    
##############################################
# core50 (to be included)
##############################################




