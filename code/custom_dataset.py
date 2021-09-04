import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
# from torchvision.io import read_image
from torch.utils.data import Dataset
from configparser import ConfigParser
import data_ops as do

# config = ConfigParser()
# config.read('../config/config.ini')

class MyImageDataset(Dataset):
    def __init__(self, config, dops1, class_to_idx=None, t_data_set = 'train', use_seed = False):
        self.data_set = config.get('data', 'dataset_name')
        self.data_dir = config.get('data', 'data_dir')
        self.root_dir = config.get('model', 'root_dir')
        self.input_type = config.get('data', 'input_type')
        self.transform_type = config.get('data', 'transform_type')
        
        # https://pytorch.org/docs/stable/notes/randomness.html
        # Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. 
        # Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.
        if use_seed:
            torch.manual_seed(config.getint('data', 'seed'))
            np.random.seed(config.getint('data', 'seed'))
        #torch.use_deterministic_algorithms(True)
        #torch.backends.cudnn.benchmark = False
        #torch.set_deterministic(True)
        #torch.backends.cudnn.deterministic = True

        self.target_transform = None
        self.t_data_set = t_data_set
        
        if t_data_set == 'train':
            self.img_dir = f'{self.data_dir}train/singles/'
            annotations_file = f'{self.root_dir}data/train_data.csv'
            self.transform = dops1.data_transforms['train']
        elif t_data_set == 'val' or t_data_set == 'val_test':
            self.img_dir = f'{self.data_dir}train/singles/' # we will make a validation set from train folder or test folder
            annotations_file = f'{self.root_dir}data/val_data.csv'
            self.transform = dops1.data_transforms['val']
        elif t_data_set == 'test':
            self.img_dir = f'{self.data_dir}test/singles/'
            annotations_file = f'{self.root_dir}data/test_data.csv'
            self.transform = dops1.data_transforms['val']

        if self.data_set == 'BioImLab':
            self.img_dir = self.img_dir[:-8]    # just remove singles/ from the end
            if t_data_set == 'val':
                self.img_dir = self.img_dir.replace('train', 'val')

        print(f'self.img_dir: {self.img_dir}')
        print(f'annotations_file: {annotations_file}')

        self.img_labels = pd.read_csv(annotations_file)
        self.class_folders = config.getboolean('data', 'class_folders')

        l_label = []
        l_index = []
        if class_to_idx == None:
            if self.input_type == 'int_str':
                l_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'x', 'y']
                l_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            elif self.input_type == 'int':
                l_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
                l_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            elif self.input_type == 'auto':
                img_classes = list(set(self.img_labels['label']))
                for i, img_class in enumerate(img_classes):
                    l_label.append(img_class)
                    l_index.append(i)

            df_label_to_index_raw = pd.DataFrame({'label': l_label, 'index': l_index})

            df_label_to_index = df_label_to_index_raw.set_index('label')
            df_index_to_label = df_label_to_index_raw.set_index('index')

            self.df_label_to_index = df_label_to_index
            self.df_index_to_label = df_index_to_label
        else:
            self.class_to_idx = class_to_idx     # e.g.
            #self.class_to_idx = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:9, 9:10, 10:11, 11:12, 12:13, 13:14, 
             #                   14:15, 15:16, 16:17, 17:18, 18:19, 19:20, 20:21, 21:22, 22:23, 23:24}

        if os.path.isfile('../data/index_to_class.csv') == False:
            self.df_index_to_label.to_csv('../data/index_to_class.csv', index=True)
            self.df_label_to_index.to_csv('../data/class_to_index.csv', index=True)

        #print(f'class_to_index:\n {self.df_label_to_index}')

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx = 0):
        '''
        NOTE: if you use PyTorch pretrained models, you need to pass a label_idx which is between 0 and (# of labels - 1), 
        e.g. if you have 5 classes, your labels passed to PyTorch training must be in [0, 1, 2, 3, 4]
        if you pass the exact class values like [1, 2, 3, 4, 5], or ['class_1', 'class_2', 'class_3', 'class_4', 'class_5']
        then you will receive a run-time error: "CUDA error: device-side assert triggered".
        Unfortunately, the error doesn't quite point the reason explicitly.
        
        Data File:
        (base) melih@arcusai-System:~/xx-dataset/data$ head train_data.csv
        image,patientID,metaphase,label,side
        pID1.005-K_11_0_.png,pID1,005,11,0
        '''
        picture = self.img_labels.iloc[idx, 0]
        if self.data_set == 'BioImLab':
            label = self.img_labels.iloc[idx, 1]
        else:
            label = self.img_labels.iloc[idx, 3]    # y_label
        #y_label = torch.tensor(label-1)   
        #label_idx = self.class_to_idx[label]         # it is a label index, not the real label
        if self.input_type != 'auto' and label not in ['x', 'y']:
            label = int(label)

#         print(f'label: {label} .. type: {type(label)}')
#         print(self.df_label_to_index.loc[str(label)][0])
        label_idx = self.df_label_to_index.loc[label]['index']  # it is a label index, not the real label
#         print(f'label_idx: {label_idx} .. type: {type(label_idx)}')

        if self.class_folders:
            #img_dir = ''.join([self.img_dir, str(label), '/'])
            img_dir = f'{self.img_dir}{str(label)}/'
        else:
            img_dir = self.img_dir

        ### returns the image with the given index               
        img_path = os.path.join(img_dir, picture)  # picture = file_name, e.g. '18 1a.bmp'
        #picture = torch.tensor(picture)
        #image = read_image(img_path)              # >> crashed the kernel, don't know the reason
        # image = io.imread(img_path)

        image_m = self.pil_loader(img_path)          # image = matrix equivalent of the picture, e.g. (224, 224, 3)

        if self.transform:
            if self.transform_type == 'albumentations':
                image_m = np.asarray(image_m)
                image = self.transform(image=image_m)
                image = image['image']
            else:
                image = self.transform(image_m)

        #print(f'picture: {picture} .. label: {label} .. label_idx: {label_idx}')
        if self.target_transform:
            label_idx = self.target_transform(label_idx)
            #picture = target_transform(picture)

        if self.t_data_set == 'test' or self.t_data_set == 'val_test':
            if self.data_set == 'BioImLab':
                sample = ['', image, label_idx, '']
            else:
                return [picture, image, label_idx, str(label)]
                #sample = (image, label_idx)
        else:
            sample = (image, label_idx)

        return sample

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
