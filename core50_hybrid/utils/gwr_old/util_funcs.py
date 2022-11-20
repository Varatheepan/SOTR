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
from random import randint
from skimage import io, transform

##########################################################
# pytorch util functions 
##########################################################

def load_checkpoint(filename) : 
    checkpoint = torch.load(filename)
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return [model_state_dict, optimizer_state_dict, epoch, loss]

def save_checkpoint(model, optimizer, epoch, loss, filename):
    print("Saving Checkpoint....")
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
#     print("model_state_dict : ", model_state_dict)
#     print("optimizer_state_dict : ", optimizer_state_dict)
#     print("Epoch : ", epoch)
#     print("filename : ", filename)
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'loss': loss
            }, filename)
    print("Saving Checkpoint Done")

#Train the network
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
def train_model(device, model, dataloaders, datasets, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training train_mode
            else:
                model.eval()   # Set model to evaluate train_mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #for inputs, labels in dataloaders[phase]:
            dataloader_phase = dataloaders[phase]
            print("Data Class : ", datasets[phase].class_idx)
            for i, data in enumerate(dataloader_phase) :
                inputs, labels = data
                # if(i==0):
                #     print("label : ", labels)
                inputs = inputs.to(device)
                labels = labels.to(device)  

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # if(i==0) : 
                    #     print("******** Loss *************", loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # if phase == 'train' : 
                #     epoch_loss = running_loss / len(train_dataset)
                #     epoch_acc = running_corrects.double() / len(train_dataset)
                # else : 
                #     epoch_loss = running_loss / len(test_dataset)
                #     epoch_acc = running_corrects.double() / len(test_dataset)
                epoch_loss = running_loss / len(datasets[phase])
                epoch_acc = running_corrects.double() / len(datasets[phase])
                

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()
        checkpoint_name = 'checkpoints/model_' + str(epoch) + '.tar'
        save_checkpoint(model, optimizer, epoch, epoch_loss, checkpoint_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def validate_model(device, model, test_loader, task=None, class_count=0, confusion_matrix=None):
    correct = 0
    total = 0
    print(next(model.parameters()).is_cuda)
    # print(device)
    # model.train()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0) :
            inputs, labels = data
            if(i==0):
                print("inputs : ", inputs)
                print("validate label : ", labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            if(task is None) : 
                outputs = model(inputs)
            else : 
                outputs = model(inputs, task)
                _, predicted = torch.max(outputs, 1)
                # print("labels : ", labels)
                # print("outputs   : ", outputs)
                # print("predicted : ", predicted)
            for k in range(predicted.size(0)):
                confusion_matrix[labels[k]][predicted[k]] += 1
            total += labels.size(0)
            # print(predicted, labels)
            correct += (predicted == labels).sum().item()
	    
	 #if(i==1) : 
#		break
    print("validate model on task : ", task, " total samples : ", total)
    print("confusion matrix : ", confusion_matrix[task])
    accuracy = 100 * (correct/total)
    print("Accuracy : ", accuracy)
    return accuracy

def validate_model_nmc(device, model, test_loader, task=None, class_count=0, confusion_matrix=None):
    correct = 0
    total = 0
    #print(next(model.parameters()).is_cuda)
    #print(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0) :
            inputs, labels = data
            if(i==0):
                print("validate label : ", labels)
            print(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            if(task is None) : 
                outputs = model(inputs)
            else : 
                outputs = model(inputs, task)
            # _, predicted = torch.max(outputs, 1)
                # predicted = model.nearest_mean_classify(outputs)
                predicted = model.predict_class(outputs)
                predicted = predicted.to(device)
                for k in range(predicted.size(0)):
                    confusion_matrix[labels[k]][predicted[k]] += 1
            total += labels.size(0)
            # print(predicted, labels)
            correct += (predicted == labels).sum().item()
    print("validate model on task : ", task, " total samples : ", total)
    print("confusion matrix : ", confusion_matrix[task])
    confusion_matrix[task] = confusion_matrix[task]/total
     #if(i==1) : 
#		break
    accuracy = 100 * (correct/total)
    print("Accuracy : ", accuracy)
    return accuracy

def get_features(device, model, dataloader, layer_name, batch_size, filename):
    model.eval()
    layer = model._modules.get(layer_name)
    num_feature_vec = len(dataloader.dataset) #batch_size * len(dataloader)
    feature_vec = torch.zeros(num_feature_vec, 512, 1, 1)
    label_vec = torch.zeros(num_feature_vec)
 
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0) :
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(inputs.size()[0])
            num_inputs = inputs.size()[0]
            feature_vec_batch = torch.zeros(num_inputs, 512, 1, 1)
            def copy_data(m, i, o):
                if(o.data.size() != feature_vec_batch.size()) : 
                    print("size mismatch, O : ",o.data.size(),  "FVec :  ", feature_vec_batch.size())      
                else : 
                    feature_vec_batch.copy_(o.data)
                
            h = layer.register_forward_hook(copy_data)
            outputs = model(inputs)   
            # print(i, " After : ", feature_vec_batch.size())
            h.remove()
            feature_vec[i*num_inputs : (i+1)*num_inputs] = feature_vec_batch
            label_vec[i*num_inputs : (i+1)*num_inputs] = labels

    feature_vec = (feature_vec.cpu()).numpy()
    label_vec   = (label_vec.cpu()).numpy()
  
    # print(feature_vec[label_vec==0]) 
    print(label_vec)

    print("Feature Vec dimensions : ", feature_vec.shape)
    print("Label Vec dimensions : ", label_vec.shape)
    arr1 = feature_vec
    arr2 = label_vec
	
    np.savez(filename, feature_vec=arr1, label_vec=arr2)
    return feature_vec, label_vec


def plot_accuracy(testname='testname', dumpname=None, 
                    cum_accuracy=0.0, 
                    classification_accuracy_av=None, classification_accuracy_by_class=None, 
                    naive_classification_accuracy_av=None,
                    class_count=0,
                    incr_train_mode=None) : 
    if(classification_accuracy_av is None or classification_accuracy_by_class is None):
        data = np.load(dumpname)
        classification_accuracy_av = data['classification_accuracy_av']
        class_accuracy_class = data['classification_accuracy_class']

    x = range(class_count)
    fig = plt.figure()
    # plt.axhline(y=cum_accuracy, color='r', linestyle='-', label='cumulative')
    plt.plot(naive_classification_accuracy_av, label='naive')
    if(incr_train_mode is None) : 
        incr_train_mode = 'gwr'
    plt.plot(classification_accuracy_av, label=incr_train_mode)
    plt.gca().set_title(('Classification Accuracy ' + testname), pad=15, fontsize=20)
    plt.gca().legend() #(bbox_to_anchor=(1.1, 1.05))
    plt.gca().set_xticks(np.arange(len(x)))
    plt.gca().yaxis.grid(True)
    plt.xlabel('Classes Encountered', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    # plt.tight_layout()
    plt_name = 'plots/' + testname + '_classification_accuracy.png'
    plt.savefig(plt_name)
    plt.show()

    #plotting classification accuracy with incremental classs training per each class
    print('plotting classification accuracy with incremental classs training per each class')
    colors = []
    for i in range(0,class_count):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    fig = plt.figure()
    x = range(0,class_count)
    for i in range(0,class_count):
        label='class'+str(i)
        print(classification_accuracy_by_class[i][-1])
        plt.plot(classification_accuracy_by_class[i], color=colors[i], label=label)
    plt.gca().set_title('Incremental Learning', pad=15, fontsize=20)
    plt.xlabel('Classes Encountered', fontsize=18)
    plt.ylabel('Classification Accuracy', fontsize=16)
    plt.gca().legend() #(bbox_to_anchor=(1.1, 1.05))
    plt.gca().set_xticks(np.arange(len(x)))
    # plt.tight_layout()
    plt_name = 'plots/' + testname + '_classification_accuracy_by_class.png'
    plt.savefig(plt_name)
    plt.show()

def plot_network(testname='testname', nodes=None, connections=None, class_count=0):
    fig = plt.figure()
    x = range(class_count)
    plt.plot(nodes, label='Nodes')
    plt.plot(connections, label='connections')
    plt.gca().set_title('GwR Nodes and Connections', pad=15, fontsize=20)
    plt.xlabel('Classes Encountered', fontsize=18)
    plt.ylabel('GwR Nodes and Connections', fontsize=16)
    plt.gca().legend()
    plt.xticks(x)
    plt.tight_layout()
    plt_name = 'plots/' + testname + '_training_nodes_connections.png'
    plt.savefig(plt_name)
    plt.show()

def plot_confusion_matrix(testname, confusion_matrix, normalize =  False, title = None, cmap=plt.cm.jet) : #cmap = plt.cm.Blues):
    # fig = plt.figure()
    classes = []
    for i in range(confusion_matrix.shape[0]):
        classes.append(i)

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion_matrix.shape[1]), yticks=np.arange(confusion_matrix.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title=testname,
        ylabel='True label',
        xlabel='Predicted label')

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    # for i in range(confusion_matrix.shape[0]):
    #     for j in range(confusion_matrix.shape[1]):
    #         ax.text(j, i, format(confusion_matrix[i, j], fmt), ha="center", va="center",
    #                 color="white" if confusion_matrix[i, j] > thresh else "black")
    # fig.tight_layout()
    # plt_name = 'plots/' + testname + '_confusion_matrix.png'
    plt_name = testname + '_confusion_matrix.png'
    plt.savefig(plt_name)
    # plt.show()

    return ax

def plot_accuracy_comparison(testname='testname', dumpname=None, 
                            cum_accuracy=0.0, 
                            classification_accuracy_arry = None,
                            label_arry = None,
                            naive_classification_accuracy_av=None,
                            ) : 
    tests = range(len(classification_accuracy_arry))
    x     = range(len(classification_accuracy_arry[0]))
    print("tests : ", tests)
    print("x     : ", x)

    #plot to demonstrate accuracy evolution
    fig = plt.figure()
    # plt.axhline(y=cum_accuracy, color='r', linestyle='-', label='cumulative')
    for test in tests :
        plt.plot(classification_accuracy_arry[test], label=label_arry[test])
    plt.gca().set_title('Classification accuracy comparison', pad=15, fontsize=20)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Encountered Classes')
    plt.title('Classification Accuracy')
    plt.gca().legend()
    plot_name = testname + '.png'
    print("plt name : ", plot_name)
    plt.savefig(plot_name)
    plt.show()

    #plot to demonstrate final accuracies
    fig = plt.figure()
    final_classification_accuracy_arry = [x[-1] for x in classification_accuracy_arry]
    plt.bar(tests, final_classification_accuracy_arry , align='center', alpha=0.5)
    plt.xticks(tests, label_arry)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Batch Size')
    plt.title('Final Accuracy comparison')
    plot_name = testname + '_final.png'
    print("plt name : ", plot_name)
    plt.savefig(plot_name)
    plt.show()

def plot_gwr_comparison(testname='testname', 
                        cum_nodes=0, cum_conns=0, 
                        nodes_arry=None, connections_arry=None,
                        label_arry=None):
    
    tests = range(len(nodes_arry))
    x     = range(len(nodes_arry[0]))

    fig   = plt.figure()
    for test in tests : 
        plt.plot(nodes_arry[test], label=label_arry[test])
    plt.gca().set_title('GwR Nodes', pad=15, fontsize=20)
    plt.gca().legend()
    plt.ylabel('Nodes')
    plt.xlabel('Encountered Classes')
    plot_name = testname + 'gwr_nodes.png'
    print("plt name : ", plot_name)
    plt.savefig(plot_name)
    plt.show()

    fig   = plt.figure()
    for test in tests : 
        plt.plot(connections_arry[test], label=label_arry[test])
    plt.gca().set_title('GwR Connections', pad=15, fontsize=20)
    plt.gca().legend()
    plt.ylabel('Connections')
    plt.xlabel('Encountered Classes')
    plot_name = testname + 'gwr_conns.png'
    print("plt name : ", plot_name)
    plt.savefig(plot_name)
    plt.show()

    fig = plt.figure()
    final_nodes_arry = [x[-1] for x in nodes_arry]
    plt.bar(tests, final_nodes_arry , align='center', alpha=0.5)
    plt.xticks(tests, label_arry)
    plt.ylabel('Nodes')
    plt.title('GwR Node comparison')
    plot_name = testname + 'gwr_nodes_final.png'
    print("plt name : ", plot_name)
    plt.savefig(plot_name)
    plt.show()

    fig = plt.figure()
    final_conns_arry = [x[-1] for x in connections_arry]
    plt.bar(tests, final_conns_arry , align='center', alpha=0.5)
    plt.xticks(tests, label_arry)
    plt.ylabel('Connections')
    plt.title('GwR Connections comparison')
    plot_name = testname + 'gwr_conns_final.png'
    print("plt name : ", plot_name)
    plt.savefig(plot_name)
    plt.show()
    
    # plt.plot(connections, label='connections')
    # plt.gca().set_title('GwR Nodes and Connections', pad=15, fontsize=20)
    # plt.xlabel('Classes Encountered', fontsize=18)
    # plt.ylabel('GwR Nodes and Connections', fontsize=16)
    # plt.gca().legend()
    # plt.xticks(x)
    # plt.tight_layout()
    # plt_name = 'plots/' + testname + '_training_nodes_connections.png'
    # plt.savefig(plt_name)
    # plt.show()


def plot_activations(taskname, activations, activation_label) : 
    activations_flat = activations.flatten() #assume numpy type
    fig = plt.figure()
    plt_title = activation_label + ' distribution'
    plt.gca().set_title(activation_label)
    plt.hist(activations_flat, bins=1000)
    plt.savefig(taskname)
    # plt.show()
    
def show_torch_image(image) : 
    plt.imshow(image.permute(1, 2, 0))
    plt.show()