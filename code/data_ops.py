import pandas as pd
import numpy as np
import os, shutil
from tqdm import tqdm
from os import listdir
import torch
from os.path import isfile, join
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# %load_ext autoreload
# %autoreload 2
# https://stackoverflow.com/questions/31237662/passing-arguments-to-a-python-script-in-a-slurm-batch-script
# https://www.tchpc.tcd.ie/node/1279
from configparser import ConfigParser

class dops:
    def __init__(self, config, class_is_int = False, use_seed=False):
        """
        When we use datasets.ImageFolder to read the data from the folder, it creates indexes for each class. To understand 
        which index belongs to which class, we need this conversion or otherwise instead of auto-loading by using the 
        datasets.ImageFolder, we should read and create the dataset by ourselves.
        """        
        self.df_index_class = None
        self.all_real_labels = {}
        self.class_is_int = class_is_int
        self.config = config
        self.data_dir = config.get('data', 'data_dir')
        self.root_dir = config.get('model', 'root_dir')
        self.model_name = config.get('model', 'name')
        self.path_index_class = f'{self.data_dir}data/index_to_class.csv'
        dict_log_level = {'error': logging.ERROR, 'info': logging.INFO, 'debug': logging.DEBUG}
        self.log_level = dict_log_level[config.get('log', 'level')]
        
        if use_seed:
            torch.manual_seed(self.config.getint('data', 'seed'))
            np.random.seed(self.config.getint('data', 'seed'))
        #torch.use_deterministic_algorithms(True)
        #torch.backends.cudnn.benchmark = False
        #torch.set_deterministic(True)
        #torch.backends.cudnn.deterministic = True

    def save_class_file(self):
        for l in self.test_data.class_to_idx:
            self.all_real_labels[dops2.test_data.class_to_idx[l]] = l
        
        self.df_index_class = pd.Series(self.all_real_labels).to_frame()
        self.df_index_class.columns = ['class']
        self.df_index_class.to_csv(self.path_index_class, index=True)
    
    def find_class(self, idx, is_int = False):
        if self.df_index_class == None:
            self.df_index_class = pd.read_csv(self.path_index_class)  # 'index_class.csv'
            self.df_index_class.drop(columns='Unnamed: 0', inplace=True)
        
        true_class = self.df_index_class.iloc[idx][0]
        
        if self.class_is_int == True:
            true_class = int(true_class)
        
        return true_class
    
    # to be reviewed ********
    def create_train_val_test(self, df_data, src_dir, tr=0.8, val=0.2, test=0.0, shuffle=False):
        csv_name = config.get('data', 'dataset_name')
        
        n_class = set(df_data['label'])  # number of distinct classes in the dataset
        self.df_data_train = pd.DataFrame({}, columns=('file_names', 'label'))
        self.df_data_val = pd.DataFrame({}, columns=('file_names', 'label'))
        
        if test != 0.0:
            self.df_data_test = pd.DataFrame({}, columns=('file_names', 'label'))
        
        if shuffle:
            df_data = df_data.sample(frac=1)
            #df = df.sample(frac=1).reset_index(drop=True)
                
        for c in n_class:
            len_c = len(df_data[df_data['label'] == c])  # nr of images belong to that class
            n_train_c = int(len_c * tr)
            n_val_c = int(len_c * val)
            n_test_c = int(len_c * test)
            
            counter = 0
            for idx, row in df_data[df_data['label'] == c].iterrows():
                if counter < n_train_c:
                    self.df_data_train.loc[idx] = row
                elif counter < n_train_c + n_val_c:
                    self.df_data_val.loc[idx] = row
                elif test != 0.0:
                    self.df_data_test.loc[idx] = row
                counter += 1                
        
        dest_dir = self.generate_dir_name(src_dir.split('/')[-2]) # src_dir = 'data/BioImLab_single_chromosomes/'
        root_dir = src_dir.split('/')[-3]
        root_dest = f'{root_dir}/{dest_dir}/'
        
        # create the directory if it is not already there
        Path(f'{root_dest}/').mkdir(parents=True, exist_ok=True)
        
        self.df_data_train.to_csv(f'{root_dest}{csv_name}_train.csv', index=False)
        self.df_data_val.to_csv(f'{root_dest}{csv_name}_val.csv', index=False)
        if test != 0.0:
            self.df_data_test.to_csv(f'{root_dest}{csv_name}_test.csv', index=False)
        
    def read_data(self, f_train = '', f_val = '', f_test = ''):
        try:
            if f_train != '':
                self.df_data_train = pd.read_csv(f_train) # 'chromosome_data_train.csv'
            if f_val != '':
                self.df_data_val = pd.read_csv(f_val)     # 'chromosome_data_val.csv'
            if f_test != '':
                self.df_data_test = pd.read_csv(f_test)   # 'chromosome_data_test.csv'
        except:
            print('Please be sure that you provided correct paths')

    def pick_data_transforms(self, data_transforms_picked, input_size):
            data_transforms_1 = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(input_size),
                    #transforms.RandomHorizontalFlip(),         # Data augmentation      
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),                    
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }                        

            data_transforms_2 = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
            
            #MEAN = [0.7763, 0.7763, 0.7763]
            #STD = [0.2744, 0.2744, 0.2744]
            MEAN = [0.6802, 0.6802, 0.6802]
            STD = [0.2875, 0.2875, 0.2875]
            #MEAN = [0.485, 0.456, 0.406]
            #STD = [0.229, 0.224, 0.225]
            SIZE = 224  # IMAGE_SIZE
            RESIZE_RATIO = 0.9
            scale = 1.1
            
            data_transforms_3 = {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(180, expand=True, center=None, fill=255),
                    ResizeLargerSide(int(RESIZE_RATIO*SIZE)),
                    RandomPad(SIZE),
                    transforms.CenterCrop(SIZE),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
                'val': transforms.Compose([
                    ResizeLargerSide(int(RESIZE_RATIO*SIZE)),
                    transforms.Pad(SIZE//2, 255),
                    transforms.CenterCrop(SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
            }
            # https://research.aimultiple.com/data-augmentation-techniques/
            # https://research.aimultiple.com/data-augmentation-deep-learning/
            # RandomHorizontalFlip, ColorJitter >> these 2 transformations don't add too much on accuracy
            data_transforms_best_TESTED = {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(),          # slightly improves to accuracy
                    # RandomRotation=180 affected the Accuracy badly. In the long run, it might reduce the uncertainty (improve the generalization) though, needs to be checked ... in other words, it can be more robust to OOD data
                    # https://bdtechtalks.com/2020/04/27/deep-learning-mode-connectivity-adversarial-attacks/
                    # RandomRotation=90 affects badly (acc 92 to 86) but not as much as 180 (acc 92 to 80)...40 > (92 to 90)
                    #transforms.RandomRotation(20, expand=True, center=None, fill=255),
                    ResizeLargerSide(int(RESIZE_RATIO*SIZE)),   # highly effective to accuracy
                    RandomPad(SIZE),                            # highly effective
                    # transforms.ColorJitter(brightness=0.5, contrast=0.5),  # reduces the accuracy 92>87
                    transforms.CenterCrop(SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
                'val': transforms.Compose([
                    ResizeLargerSide(int(RESIZE_RATIO*SIZE)),
                    transforms.Pad(SIZE//2, 255),
                    transforms.CenterCrop(SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
            }
            data_transforms_best = {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(),          # slightly improves to accuracy
                    # RandomRotation=180 affected the Accuracy badly. In the long run, it might reduce the uncertainty (improve the generalization) 
                    #though, needs to be checked ... in other words, it can be more robust to OOD data
                    # https://bdtechtalks.com/2020/04/27/deep-learning-mode-connectivity-adversarial-attacks/
                    # RandomRotation=90 affects badly (acc 92 to 86) but not as much as 180 (acc 92 to 80)...40 > (92 to 90)
                    #transforms.RandomRotation(20, expand=True, center=None, fill=255),
                    ResizeLargerSide(int(RESIZE_RATIO*SIZE)),   # highly effective to accuracy
                    RandomPad(SIZE),                            # highly effective
                    # transforms.ColorJitter(brightness=0.5, contrast=0.5),  # reduces the accuracy 92>87
                    transforms.CenterCrop(SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
                'val': transforms.Compose([
                    ResizeLargerSide(int(RESIZE_RATIO*SIZE)),
                    transforms.Pad(SIZE//2, 255),
                    transforms.CenterCrop(SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
            }
            # ResizeLargerSide >> this one gives the most positive impact on Accuracy
            data_transforms_32 = {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    ResizeLargerSide(int(RESIZE_RATIO*SIZE)),
                    transforms.CenterCrop(SIZE),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
                'val': transforms.Compose([
                    ResizeLargerSide(int(RESIZE_RATIO*SIZE)),
                    transforms.Pad(SIZE//2, 255),
                    transforms.CenterCrop(SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
            }
            # RandomPad >> this one gives the 2nd most positive impact on Accuracy
            data_transforms_33 = {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    RandomPad(SIZE),
                    transforms.CenterCrop(SIZE),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
                'val': transforms.Compose([
                    ResizeLargerSide(int(RESIZE_RATIO*SIZE)),
                    transforms.Pad(SIZE//2, 255),
                    transforms.CenterCrop(SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ]),
            }
            
            data_transforms_4 = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.RandomResizedCrop(input_size),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),         # Data augmentation   
                    #transforms.RandomRotation(180, expand=True, center=None, fill=255),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    #transforms.Pad(input_size//2, input_size),
                    transforms.CenterCrop(224),                    
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
            
            # IF I comment out some part of augmentations, in other words, if I don't use the same augmentations for both TRAIN and VAL/TEST, then TEST accuracy drops significantly,
            # the solution overfits to TRAINing augmentation
            data_transforms_albumentations1 = {
                    'train': A.Compose([
                        A.LongestMaxSize(max_size=int(SIZE * scale)),
                            A.PadIfNeeded(
                                min_height=int(SIZE * scale),
                                min_width=int(SIZE * scale),
                                border_mode=cv2.BORDER_CONSTANT,
                            ),
                            A.RandomCrop(width=SIZE, height=SIZE),
                            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
                            A.OneOf(
                                [
                                    A.ShiftScaleRotate(
                                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                                    ),
                                    A.IAAAffine(shear=15, p=0.5, mode="constant"),
                                ],
                                p=1.0,
                            ),
                            A.HorizontalFlip(p=0.5),
                            A.Blur(p=0.1),
                            A.CLAHE(p=0.1),
                            A.Posterize(p=0.1),
                            A.ToGray(p=0.1),
                            A.ChannelShuffle(p=0.05),
                            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                            ToTensorV2(),
                        ]),
                    'val': A.Compose([
                        A.LongestMaxSize(max_size=int(SIZE * scale)),
                            A.PadIfNeeded(
                                min_height=int(SIZE * scale),
                                min_width=int(SIZE * scale),
                                border_mode=cv2.BORDER_CONSTANT,
                            ),
                            A.RandomCrop(width=SIZE, height=SIZE),
                            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
                            #A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1.0),  # apply ColorJitter each time
                            A.OneOf(
                                [
                                    A.ShiftScaleRotate(
                                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                                    ),
                                    A.IAAAffine(shear=15, p=0.5, mode="constant"),
                                ],
                                p=1.0,
                            ),
                            A.HorizontalFlip(p=0.5),
                            A.Blur(p=0.1),
                            A.CLAHE(p=0.1),
                            A.Posterize(p=0.1),
                            A.ToGray(p=0.1),
                            A.ChannelShuffle(p=0.05),
                            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                            ToTensorV2(),
                        ]),
            }

            data_transforms_ch1D = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize([0.485], [0.229])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize([0.485], [0.229])
                ]),
            }

            if data_transforms_picked == 'data_transforms_1':
                self.data_transforms = data_transforms_1
            elif data_transforms_picked == 'data_transforms_2':
                self.data_transforms = data_transforms_2
            elif data_transforms_picked == 'data_transforms_ch1D':
                self.data_transforms = data_transforms_ch1D
            elif data_transforms_picked == 'data_transforms_3':
                self.data_transforms = data_transforms_3
            elif data_transforms_picked == 'data_transforms_best':
                self.data_transforms = data_transforms_best
            elif data_transforms_picked == 'data_transforms_32':
                self.data_transforms = data_transforms_32
            elif data_transforms_picked == 'data_transforms_33':
                self.data_transforms = data_transforms_33
            elif data_transforms_picked == 'data_transforms_albumentations1':
                self.data_transforms = data_transforms_albumentations1

    def load_data_sets(self, data_set_picked, batch_size, ds_params = None):
        # ds_params = [root, train, val, download]
        # ds_params = ['data', True, False, False]
        if data_set_picked == 'FashionMNIST':
            self.training_data = datasets.FashionMNIST(
                root=ds_params[0],                              #"data",
                train=ds_params[1],                             #True,
                download=ds_params[3],                          #False,
            #     transform=transforms.ToTensor(),
                transform=self.data_transforms['train'], #data_transforms_1D['train']
            )

            # Download test data from open datasets.
            self.val_data = datasets.FashionMNIST(
                root=ds_params[0],                              #"data",
                train=ds_params[2],                             #False,
                download=ds_params[3],                          #False,
            #     transform=transforms.ToTensor(),
                transform=self.data_transforms['val'],   #data_transforms_1D['val']
            )

        self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size)
        self.val_dataloader = DataLoader(self.val_data, batch_size=batch_size)

        self.dataloaders_dict = {'train': self.train_dataloader, 'val': self.val_dataloader}

    def load_test_data(self, data_set_picked, batch_size, ds_params = None):
        # ds_params = [root, download]
        # ds_params = ['data', False]
        if data_set_picked == 'FashionMNIST':
            # Download test data from open datasets.
            self.test_data = datasets.FashionMNIST(
                root=ds_params[0],                              #"data",
                train=False,                                    #False,
                download=ds_params[1],                          #False,
            #     transform=transforms.ToTensor(),
                transform=self.data_transforms['val'],   #data_transforms_1D['val']
            )

        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size)

        # self.dataloaders_dict = {'test': self.test_dataloader}

    def generate_file_name(self, dir_name, prefix = 'resnet18', suffix = 'MNIST'):
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%Y_%H.%M.%S")
        print("date and time =", dt_string)

        file_name = f'{dir_name}/{prefix}_{dt_string}_{suffix}'  
        
        return file_name
    
    def generate_dir_name(dir_name):
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%Y")

        dir_name = f'{dir_name}_{dt_string}'

        return dir_name
        
    def create_csv(self, train_val=[0.8, 0.2]):
        # Stratified sampling  >>>> CHECK THAT ONE
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(data, train_y, test_size=ratio, random_state=10, stratify=train_y)
        dir_train = f'{self.dir_data}train/singles/'
        dir_test = f'{self.dir_data}test/singles/'
        train_files = os.listdir(dir_train)
        test_files = os.listdir(dir_test)                
        
        for i, files in tqdm(enumerate([train_files, test_files])):
            l_patientID = []
            l_metaphase = []
            l_chrmsm_class = []
            l_chrmsm_side = []

            for f in tqdm(files):
                [patientID, chrmsm_info, _] = f.split('.')
                [metaphase, chrmsm_info2] = chrmsm_info.split('-')
                [_, chrmsm_class, side, _] = chrmsm_info2.split('_')

                l_patientID.append(patientID)
                l_metaphase.append(metaphase)
                l_chrmsm_class.append(chrmsm_class)
                l_chrmsm_side.append(side)
        
            print(f'l_patientID: {len(l_patientID)} .. l_metaphase: {len(l_metaphase)} .. l_chrmsm_class: {len(l_chrmsm_class)}')
            print(f'l_chrmsm_side: {len(l_chrmsm_side)} .. files: {len(files)}')
            df_data = pd.DataFrame({'image': files, 'patientID': l_patientID, 
                         'metaphase': l_metaphase, 'label': l_chrmsm_class, 'side': l_chrmsm_side})
            
            if i == 1:
                print(f'processing Test data ... {datetime.now()}')
                df_data.to_csv(f'{self.root_dir}data/test_data.csv', index=False)
            else:
                print(f'processing Train data ... {datetime.now()}')
                l_class = list(set(df_data['label']))
                df_data_train = pd.DataFrame({}, columns=('image', 'patientID', 'metaphase', 'label', 'side'))
                df_data_val = pd.DataFrame({}, columns=('image', 'patientID', 'metaphase', 'label', 'side'))
                
                for c in tqdm(l_class):
                    print(f'class {c} is in progress: {datetime.now()}')
                    len_c = len(df_data[df_data['label'] == c])  # nr of images belong to that class
                    n_train_c = int(len_c * train_val[0])
                    n_val_c = len_c - n_train_c                  #int(len_c * train_val[1])

                    counter = 0
                    for idx, row in df_data[df_data['label'] == c].iterrows():
                        if counter < n_train_c:
                            df_data_train.loc[idx] = row
                        else:                                     #elif counter < n_train_c + n_val_c:
                            df_data_val.loc[idx] = row
                        counter += 1
                        
                df_data_train.to_csv(f'{self.root_dir}data/train_data.csv', index=False)
                df_data_val.to_csv(f'{self.root_dir}data/val_data.csv', index=False)
                
#                 filename='/Users/melih/Documents/Masters/KTH_DASE-ICT Innovation_Data Science/___Master Thesis/PyTorch_Exercise/arkusai/logs/dev.log'
    def get_logger(self, name):
        log_path = self.generate_file_name(f'{self.root_dir}logs/', prefix = self.model_name, suffix = '.log')
        log_format = '%(asctime)s  %(name)8s  %(levelname)5s  %(message)s'
        logging.basicConfig(level=self.log_level,
                            format=log_format,
                            filename=log_path,
                            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(self.log_level)
        console.setFormatter(logging.Formatter(log_format))
        logging.getLogger(name).addHandler(console)
        return logging.getLogger(name)
        
class ResizeLargerSide(object):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    def __call__(self, img):
        w, h = img.size
        if h < w:
            new_w = self.output_size
            new_h = int(h*(self.output_size/w))
        elif w < h:
            new_h = self.output_size
            new_w = int(w*(self.output_size/h))
        else:  # w == h
            new_w = self.output_size
            new_h = self.output_size
        resized_img = img.resize((new_w, new_h))
        return resized_img

class RandomPad(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, image):
        w, h = image.size
        new_h, new_w = self.output_size
        color = (255, 255, 255)
        pad_top, pad_bottom, pad_l, pad_r = 0, 0, 0, 0
        if h < new_h:  # need to pad height
            pad_top = np.random.randint(0, new_h - h)
        if w < new_w:  # need to pad width
            pad_l = np.random.randint(0, new_w - w)
        padded_img = Image.new(image.mode, (new_w, new_h), color)
        padded_img.paste(image, (pad_l, pad_top))
        return padded_img
    
