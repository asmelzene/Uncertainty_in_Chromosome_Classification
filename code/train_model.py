# from __future__ import print_function
# from __future__ import divisiondivision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

from humanfriendly import format_timespan
from datetime import datetime
import os
import pickle
from configparser import ConfigParser
import argparse
# import copy
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
# from pathlib import Path

#import preprocess_training_data as ptd
# import file_ops as fops
import data_ops as do
#import pytorch_pretrained_models_CLR as ppm
import pytorch_pretrained_models as ppm
import custom_dataset as cds

def main():    
    ctm = c_train_model(config_path)   
    
    if ctm.partition_needed:
        ctm.generate_directories()
            
    ctm.run_data_operations()

    ctm.train_model()

class c_train_model:
    def __init__(self, config_path):
        config = ConfigParser()
        #config.read('../config/config.ini')
        config.read(config_path)
        self.config = config
        
        torch.manual_seed(self.config.getint('data', 'seed'))
        np.random.seed(self.config.getint('data', 'seed'))
        #torch.use_deterministic_algorithms(True)
        #torch.backends.cudnn.benchmark = False
        #torch.set_deterministic(True)
        #torch.backends.cudnn.deterministic = True

        dops1 = do.dops(self.config, use_seed = True)
        self.dops1 = dops1

        self.logger = dops1.get_logger('train')

        for section in self.config.sections():
            self.logger.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> section: {section}')
            for (key, val) in self.config.items(section):
                self.logger.info(f'{key}: {val}')
        
        ### ******************* SET the Configuration Parameters ********************** ###
        ### *************************************************************************** ###
        ### for explanations related to the config parameters, check 'config_explanations.py' file
        self.root_dir = config.get('model', 'root_dir')
        self.model_name = config.get('model', 'name')
        self.num_classes = config.getint('model', 'num_classes')
        self.batch_size = config.getint('model', 'batch_size')
        self.num_epochs = config.getint('model', 'num_epochs')
        self.single_channel = config.getboolean('model', 'single_channel')  
        self.feature_extract = config.getboolean('model', 'feature_extract')
        self.use_pretrained = config.getboolean('model', 'use_pretrained') 
        
        self.data_dir = config.get('data', 'data_dir')        
        self.data_transform = config.get('data', 'transform')
        self.csv_name_pre = config.get('data', 'dataset_name')
        self.partition_needed = config.getboolean('data', 'partition_needed')         
        self.custom_dataset = config.getboolean('data', 'custom_dataset')
        self.class_folders = config.getboolean('data', 'class_folders')
        self.shuffle = config.getboolean('data', 'shuffle')
        self.p_requires_grad = config.getboolean('model', 'p_requires_grad')
        self.show_params = config.getboolean('model', 'show_params')
        self.use_prev_model = config.getboolean('model', 'use_prev_model')        
        self.pre_trained_file = config.get('model', 'prev_model') 
        self.train_csv = f'{self.root_dir}data/train_data.csv'
        self.val_csv = f'{self.root_dir}data/val_data.csv'
        
        self.optim_selected = config.get('optimization', 'optimizer')
        self.lr = config.getfloat('optimization', 'learning_rate')
        self.weight_decay = config.getfloat('optimization', 'weight_decay')
        self.momentum = config.getfloat('optimization', 'momentum')
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.start_date = datetime.now()
        
        self.logger.info(f'Starting: {self.start_date}')

    def generate_directories(self):
        # Generate Directories for Train/Val/Test If Needed
        prep_train = ptd.preprocess()
        prep_train.ratios = [0.8, 0.1, 0.1] # train, val, tests
        prep_train.raw_data_meta = 'data/chromosome_data.csv'
        prep_train.raw_src_dir = 'data/BioImLab_single_chromosomes/'
        prep_train.csv_name_pre = self.csv_name_pre
        self.data_dir = prep_train.preprocess_training()
        
    def run_data_operations(self):
        data_dir, csv_name_pre, class_folders, batch_size, shuffle = self.data_dir, self.csv_name_pre, self.class_folders, self.batch_size, self.shuffle
        
        self.pre_tr_model = ppm.pt_model(self.config, training = True, use_seed = True)
        self.model_ft, self.input_size = self.pre_tr_model.initialize()
        
        self.dops1.pick_data_transforms(self.data_transform, self.pre_tr_model.input_size)
        self.logger.info(f'Data transforms are picked: {datetime.now()}')
        
        ### ******************* Continue with previous Weights ********************************* ###
        ### ************************************************************************************ ###        
        if self.use_prev_model == True:
             self.model_ft.load_state_dict(torch.load(self.pre_trained_file, map_location=self.device))
                        
        ### if we want to continue training from where we left to improve the previous accuracy 
        # model_ft.load_state_dict(torch.load('models/squeezenet1_0_15.04.2021_22.02.26_BioImLab_50epochs_0.8284.pth'))
        # model_ft.load_state_dict(torch.load('models/resnet18_26epochs_0.7413.weights', map_location=self.device))
        
        self.logger.info(f'model initialization is done: {datetime.now()}')
        
        ### ******************* Create Datasets ************************************************ ###
        ### ************************************************************************************ ###
        if self.custom_dataset:
            # ### >>> one way of creating datasets > by using your own-custom dataset:                        
            train_dataset = cds.MyImageDataset(self.config, self.dops1, t_data_set = 'train', use_seed = True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=self.shuffle)

            val_dataset = cds.MyImageDataset(self.config, self.dops1, t_data_set = 'val', use_seed = True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=self.shuffle)

            self.datasets_dict = {'train': train_dataset, 'val': val_dataset}
            self.dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}
        else:
            ##### >>> from PyTorch official tutorial:
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                            dops1.data_transforms[x]) for x in ['train', 'val']}

            self.datasets_dict = image_datasets
            # dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
            #                                 batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
            self.dataloaders_dict = {x: DataLoader(image_datasets[x], 
                                            batch_size=batch_size, shuffle=self.shuffle) for x in ['train', 'val']}
            ## image_datasets['train'].class_to_idx  # >> gives class -> index mapping
            ## dops1.save_class_file()
        
        self.logger.info(f'Dataloaders dictionary is ready: {datetime.now()}')
        
    def train_model(self):
        #### ******** TRY CROSS-VALIDATION **********
        params_to_update = self.pre_tr_model.which_params(p_requires_grad=self.p_requires_grad, show_params=self.show_params)
        # Observe that all parameters are being optimized
        if self.optim_selected == 'SGD':
            self.optimizer_ft = optim.SGD(params_to_update, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optim_selected == 'Adam':
            self.optimizer_ft = optim.Adam(params_to_update)
        #self.optimizer_ft = optim.SGD(params_to_update, lr=0.1, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()           # Setup the loss fxn
        ### NOTE: Consider saving&loading of the previous optimizer, too (also, learning rate?) ... 
        ### torch.save(self.optimizer_ft, file_name)
        ### self.optimizer_ft.load_state_dict(file_name, map_location=device)
        ### If we don't do this then it will just have learning rate of old checkpoint
        ### and it will lead to many hours of debugging \:  >> haven't tried/checked below
        #for param_group in optimizer.param_groups:
        #    param_group["lr"] = lr

        self.logger.info(f'Training the model: {datetime.now()}')
#         if self.custom_dataset:
#             self.model_ft, self.result_hist, self.best_preds, self.best_labels, self.best_outputs, self.best_acc = \
#           self.pre_tr_model.train_custom(self.model_ft, self.dataloaders_dict, self.criterion, self.optimizer_ft, \
#           num_epochs=self.num_epochs, is_inception=(self.model_name=="inception"))
#         else:
#             self.model_ft, self.result_hist, self.best_preds, self.best_labels, self.best_outputs, self.best_acc = \
#           self.pre_tr_model.train(self.model_ft, self.dataloaders_dict, self.criterion, self.optimizer_ft, \
#           num_epochs=self.num_epochs, is_inception=(self.model_name=="inception"))
        self.model_ft, self.result_hist, self.best_preds, self.best_labels, self.best_outputs, self.best_acc = \
          self.pre_tr_model.train(self.model_ft, self.dataloaders_dict, self.criterion, self.optimizer_ft, \
          num_epochs=self.num_epochs, is_inception=(self.model_name=="inception"))
        
        self.logger.info(f'Saving the best weights of the model: {datetime.now()}')
        self.pre_tr_model.save_model_weights(self.model_name, 
        f'{self.csv_name_pre}_{self.num_epochs}epochs_{np.round(self.best_acc.detach().cpu().numpy(), 4)}')
        
        # generate_file_name(dir_name, prefix = 'resnet18', suffix = 'MNIST')
        file_results = self.dops1.generate_file_name(f'{self.root_dir}results', prefix=self.model_name, suffix='result_hist.pkl')
        self.logger.info(f'file_results: {file_results}')
        self.result_hist.to_pickle(file_results)
        self.logger.info(f'results save to: {file_results}')
        
        # with open('results/resnet18_22.04.2021_16.36.39_result_hist.pkl', 'rb') as fp:
        #   self.result_hist = pickle.load(fp)
        # result_hist['train_acc'] = result_hist['train_acc'].apply(lambda x: x.detach().cpu().numpy())
        # result_hist['val_acc'] = result_hist['val_acc'].apply(lambda x: x.detach().cpu().numpy())

        val_acc_hist = self.result_hist['val_acc']
        self.logger.info(f'Validation accuracy history:\n{val_acc_hist}')
        end_date=datetime.now()
        self.logger.info(f'ALL done: {end_date}')

        t_delta=end_date-self.start_date
        self.logger.info(format_timespan(t_delta.seconds))

    def run_all(self):
        if self.partition_needed:
            self.generate_directories()
            
        self.run_data_operations()
        
        self.train_model()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/config.ini', help='which config file to use')
    
    opt = parser.parse_args()
    config_path = opt.config
    
    main()
