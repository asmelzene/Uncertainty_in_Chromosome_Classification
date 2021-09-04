# from __future__ import print_function
# from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import models
import copy
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from datetime import datetime
import pickle
from configparser import ConfigParser
import data_ops as dops

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# import autoreload
# %load_ext autoreload
# %autoreload 2

class pt_model:
    def __init__(self, config, m_name='densenet121', training = False, use_seed = False):
        self.config = config
        
        dops1 = dops.dops(self.config)
        self.dops1 = dops1
        self.logger = dops1.get_logger('train_ppt')    
        
        if use_seed:
            torch.manual_seed(self.config.getint('data', 'seed'))
            np.random.seed(self.config.getint('data', 'seed'))
        
        #torch.use_deterministic_algorithms(True)
        #torch.backends.cudnn.benchmark = False
        #torch.set_deterministic(True)
        #torch.backends.cudnn.deterministic = True
        
        # >>> config for BOTH TRAINING and TEST
        # config = [data_dir, model_name, num_classes, batch_size, single_channel, use_pretrained, num_epochs, feature_extract]
        # Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
        self.data_dir = config.get('data', 'data_dir')
        self.model_type = config.get('model', 'type')
        if self.model_type == 'mix-multi':
            self.model_name = m_name
        else:
            self.model_name = config.get('model', 'name')      # e.g. [resnet, alexnet, vgg, squeezenet, densenet, inception]
        self.num_classes = config.getint('model', 'num_classes') # Number of classes in the dataset
        # Batch size for training (change depending on how much memory you have)
        self.batch_size = config.getint('model', 'batch_size')  # e.g. 64
        self.single_channel = config.getboolean('model', 'single_channel')
        self.use_pretrained = config.getboolean('model', 'use_pretrained')  # if False, train from Scratch, else use pth
        self.root_dir = config.get('model', 'root_dir')
        self.drop_rate = config.getfloat('model', 'drop_rate')
        self.training = training

        # >>> config Needed ONLY for TRAINING
        self.num_epochs = config.getint('model', 'num_epochs')  # Number of epochs to train for, e.g. 100
        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params >> we usually use False in our tests
        self.feature_extract = config.getboolean('model', 'feature_extract')
        self.custom_dataset = config.getboolean('data', 'custom_dataset')   # e.g. True
        
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        #print(f'self.device: {self.device}')
        #logging1.info(f'self.device: {self.device}')     
        self.logger.info(f'self.device: {self.device}')

    def enable_dropout_Densenet(self, model, drop_rate = 0.1):
        for m in model.modules():
            if m.__class__.__name__.startswith('_DenseBlock'):
                m.drop_rate = drop_rate

    def initialize(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        #print("Initializing Datasets and Dataloaders...")
        self.model_ft = None
        self.input_size = 0

        if self.model_name == "resnet18":
            """ Resnet18
            """
            self.model_ft = models.resnet18(pretrained=self.use_pretrained)
            if self.training:
                self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
                
            self.num_ftrs = self.model_ft.fc.in_features
            # Modify ResNet for the "number of outputs" (original has 1000 classes, we modify it to 24 classes)
            self.model_ft.fc = nn.Linear(self.num_ftrs, self.num_classes)
            # Modify ResNet for the "single channel grayscale"
            if self.single_channel:
                self.model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.input_size = 224

        elif self.model_name == "alexnet":
            """ Alexnet
            """
            self.model_ft = models.alexnet(pretrained=self.use_pretrained)
            if self.training:
                self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(self.num_ftrs,self.num_classes)
            self.input_size = 224

        elif self.model_name == "VGG11_bn":
            """ VGG11_bn
            """
            self.model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            if self.training:
                self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(self.num_ftrs,self.num_classes)
            self.input_size = 224

        elif self.model_name == "squeezenet1_0":
            """ Squeezenet
            """
            self.model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            if self.training:
                self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            
            self.model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
            self.model_ft.num_classes = self.num_classes
            input_size = 224

        elif self.model_name == "densenet121":
            """ Densenet
            """
            self.model_ft = models.densenet121(pretrained=self.use_pretrained)
            if self.training:
                self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.num_ftrs = self.model_ft.classifier.in_features
            self.model_ft.classifier = nn.Linear(self.num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "inception_v3":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.model_ft = models.inception_v3(pretrained=self.use_pretrained)
            if self.training:
                self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            # Handle the auxilary net
            self.num_ftrs = self.model_ft.AuxLogits.fc.in_features
            self.model_ft.AuxLogits.fc = nn.Linear(self.num_ftrs, self.num_classes)
            # Handle the primary net
            self.num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(self.num_ftrs,self.num_classes)
            self.input_size = 299
            self.use_CLR = False

        else:
            print("Invalid model name, exiting...")
            self.logger.info("Invalid model name, exiting...")
            exit()
        
        try:
            self.model_ft = self.model_ft.to(self.device) # Send the model to GPUx
            self.logger.info(f'model in use: {self.model_name}')
        except:
            print("Model could not be loaded to the device, exiting...")
            self.logger.info("Model could not be loaded to the device, exiting...")
            exit()

        return self.model_ft, self.input_size

    def which_params(self, p_requires_grad = True, show_params = False):
        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad is True.
        params_to_update = self.model_ft.parameters()
        if show_params:
            print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name,param in tqdm(self.model_ft.named_parameters()):
                if param.requires_grad == p_requires_grad:    # True, if training
                    params_to_update.append(param)
                    if show_params:
                        print("\t",name)
                        pass
        else:
            for name,param in tqdm(self.model_ft.named_parameters()):
                if param.requires_grad == p_requires_grad:
                    if show_params:
                        print("\t",name)
                        pass
        
        return params_to_update

    def train(self, model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []
        val_loss_history = []
        train_acc_history = []
        train_loss_history = []
        optim_selected = self.config.get('optimization', 'optimizer')
        use_CLR = self.config.getboolean('optimization', 'use_clr')
        if optim_selected == 'SGD' and use_CLR:
            l_LR = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        flag_acc = 0.0
        best_outputs = []
        best_preds = []
        best_labels = []
        
        stop_training = False

        #https://pytorch.org/docs/stable/optim.html
        #print(f'use_CLR: {use_CLR}')
        if optim_selected == 'SGD' and use_CLR:            
            # step_size_up=len(dataloaders['train'])  
            # >> 1st epoch ends up at the top of the triangle, 2nd epoch down and so on
            # e.g. 100 epochs = 50 cycles ... if 2*len then 100 epochs = 25 cycles
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.config.getfloat('optimization', 'base_lr'), max_lr=self.config.getfloat('optimization', 'max_lr'), step_size_up=2*len(dataloaders['train']), mode=self.config.get('optimization', 'cycle_mode'))
        #scheduler.step()

        for epoch in tqdm(range(num_epochs)):
#             print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#             print('-' * 10)
            self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            self.logger.info('-' * 10)

            if stop_training == True:
                break

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if self.drop_rate > 0 and self.drop_rate < 1:
                        print('dropout is being set')
                        self.enable_dropout_Densenet(model, drop_rate = self.drop_rate)
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                # (picture, image, label_idx, label)
                #for (_, inputs, labels, _) in tqdm(dataloaders[phase]):
                for (inputs, labels) in tqdm(dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients  # do we need to zero them?? #### ********** MEL
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            if optim_selected == 'SGD' and use_CLR:
                                scheduler.step()
                                l_LR.append(scheduler.get_last_lr()[0])

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                self.logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_outputs = outputs
                    best_preds = preds
                    best_labels = labels
                    
                    if best_acc > self.config.getfloat('threshold', 'acc_save') or (epoch_acc - flag_acc) > self.config.getfloat('threshold', 'acc_diff_save'):
                        flag_acc = epoch_acc
                        torch.save(best_model_wts, f'{self.root_dir}models/{self.model_name}_{epoch}epochs_{np.round(best_acc.detach().cpu().numpy(), 4)}.weights')
                if phase == 'val':
                    val_acc_history.append(np.round(epoch_acc.detach().cpu().numpy(), 4))
                    #val_loss_history.append(np.round(epoch_loss.detach().cpu().numpy(), 4))
                    val_loss_history.append(np.round(epoch_loss, 4))
                elif phase == 'train':
                    train_acc_history.append(np.round(epoch_acc.detach().cpu().numpy(), 4))
                    #train_loss_history.append(np.round(epoch_loss.detach().cpu().numpy(), 4))
                    train_loss_history.append(np.round(epoch_loss, 4))
                    
                if epoch_acc >= self.config.getfloat('threshold', 'acc_stop') or epoch_loss < self.config.getfloat('threshold', 'loss_stop'):
                    stop_training = True
                    
            print()

        time_elapsed = time.time() - since
#         print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#         print('Best val Acc: {:4f}'.format(best_acc))
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        self.model_ft = model
        
        result_hist = pd.DataFrame({'train_acc': train_acc_history, 'train_loss': train_loss_history, 'val_acc': val_acc_history, 'val_loss': val_loss_history})

        # https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling #>>plot LR
        if optim_selected == 'SGD' and use_CLR:
            f_name_LR = self.dops1.generate_file_name(f'{self.root_dir}results', prefix = f'{self.model_name}', suffix = '_LR.pkl')
            df_LR = pd.DataFrame({'LR':l_LR})
            #df_LR.to_pickle('/home/melih/arkusai/results/LR.pkl')
            df_LR.to_pickle(f_name_LR)
            print(f'{f_name_LR} saved')

        return model, result_hist, best_preds, best_labels, best_outputs, best_acc
    
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in tqdm(model.parameters()):
                param.requires_grad = False

    # ************ run_model is not in use, will be revised!
    def run_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model_name, num_classes, feature_extract, use_pretrained
        self.model_ft, self.input_size = initialize_model(self.model_name, self.num_classes, 
            self.feature_extract, self.use_pretrained)
        self.model_ft = self.model_ft.to(device)

        params_to_update = model_ft.parameters()

        #print("Params to learn:")

        if self.feature_extract:
            params_to_update = []
            for name,param in tqdm(self.model_ft.named_parameters()):
                if param.requires_grad == True:
                    self.params_to_update.append(param)
                # print("\t",name)
                pass
        else:
            for name,param in tqdm(self.model_ft.named_parameters()):
                if param.requires_grad == True:
                #   print("\t",name)
                    pass

        # Observe that all parameters are being optimized
        self.optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        self.criterion = nn.CrossEntropyLoss()

        self.model_ft, self.hist, self.best_preds, self.best_labels, self.best_outputs = train_model(self.model_ft, 
            self.dataloaders_dict, self.criterion, self.optimizer_ft, num_epochs=self.num_epochs, 
            is_inception=(self.model_name=="inception"))

    def save_model_weights(self, prefix = 'resnet18', suffix = 'MNIST'):
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%Y_%H.%M.%S")
#         print("date and time =", dt_string)

        file_name = f'{self.root_dir}models/{prefix}_{dt_string}_{suffix}.pth'
        torch.save(self.model_ft.state_dict(), file_name)
#         print("Saved PyTorch Model State to " + file_name)
        self.logger.info("Saved PyTorch Model State to " + file_name)
        """
        # to use the model again:
        model = models.resnet18(pretrained=use_pretrained)
        model.load_state_dict(torch.load("model.pth"))
        """
