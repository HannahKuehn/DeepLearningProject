#!/usr/bin/env python
# coding: utf-8

# In[8]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# 
# Finetuning Torchvision Models
# =============================
# 
# **Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__
# 
# 
# 

# In this tutorial we will take a deeper look at how to finetune and
# feature extract the `torchvision
# models <https://pytorch.org/docs/stable/torchvision/models.html>`__, all
# of which have been pretrained on the 1000-class Imagenet dataset. This
# tutorial will give an indepth look at how to work with several modern
# CNN architectures, and will build an intuition for finetuning any
# PyTorch model. Since each model architecture is different, there is no
# boilerplate finetuning code that will work in all scenarios. Rather, the
# researcher must look at the existing architecture and make custom
# adjustments for each model.
# 
# In this document we will perform two types of transfer learning:
# finetuning and feature extraction. In **finetuning**, we start with a
# pretrained model and update *all* of the model’s parameters for our new
# task, in essence retraining the whole model. In **feature extraction**,
# we start with a pretrained model and only update the final layer weights
# from which we derive predictions. It is called feature extraction
# because we use the pretrained CNN as a fixed feature-extractor, and only
# change the output layer. For more technical information about transfer
# learning see `here <https://cs231n.github.io/transfer-learning/>`__ and
# `here <https://ruder.io/transfer-learning/>`__.
# 
# In general both transfer learning methods follow the same few steps:
# 
# -  Initialize the pretrained model
# -  Reshape the final layer(s) to have the same number of outputs as the
#    number of classes in the new dataset
# -  Define for the optimization algorithm which parameters we want to
#    update during training
# -  Run the training step
# 
# 
# 

# In[9]:


from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "sortedimages"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 37

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for 
num_epochs = 10

veryfinetune = True

finetunenames = ["layer4.1.bn2", "layer4.1.conv2"]

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = True

def test_model(model, dataloaders, criterion, optimizer):

            # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

                # zero the parameter gradients
        optimizer.zero_grad()

                # forward
                # track history if only in train
                
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

    
        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double()/len(dataloaders['test'].dataset)
    return acc

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                   

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for name,param in model.named_parameters():
            if veryfinetune:
                for layername in finetunenames:
                    if name.startswith(layername):
                        param.requires_grad = True
                        break
                    else:
                        param.requires_grad = False
            else:
                param.requires_grad = False


def printLayers(model_ft):
    for name,param in model_ft.named_parameters():
        print("\t",name)

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    
    ########## pretrained model #############
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size

# Initialize the model for this run

# Train and evaluate

if __name__ == "__main__":
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    print(model_ft)
    ######## data transformation and augmentation #########
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            #transforms.GaussianBlur(1.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing()
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val','test']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val','test']}



    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(torch.cuda.is_available())

    model_ft = model_ft.to(device)

    print("All parameters:")
    printLayers(model_ft)
    params_to_update = model_ft.parameters()
    bn_params = []
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                if "bn"  in name:
                    bn_params.append(param)
                    print("\t bn",name)
                else:
                    params_to_update.append(param)
                    print("\t other",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
                if "bn"  in name:
                    bn_params.append(param)
   
    #### OPTIMIZER AND LEARNING RATES #########

   # optimizer_ft = optim.Adam(params_to_update, lr=0.001)
    optimizer_ft = optim.SGD([
              {'params': bn_params},
                {'params': params_to_update, 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)

   # optimizer_ft = optim.Adagrad([
      #          {'params': bn_params},
        #        {'params': params_to_update, 'lr': 1e-3}
        #    ], lr=1e-2)



    ####### LOSS FUNCTION ############
    criterion = nn.MultiMarginLoss()
    #criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=False)
    model_ft.eval()
    print(test_model(model_ft,dataloaders_dict,criterion,optimizer_ft).item())


