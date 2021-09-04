import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import image
import data_ops as do
import custom_dataset as cds
import argparse
from configparser import ConfigParser
from tqdm import tqdm
from torchvision import transforms
from numpy import asarray
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../config/config.ini', help='which config file to use')
parser.add_argument('--opt', type=str, default='train')

opt = parser.parse_args()
config_path = opt.config
opt_type = opt.opt
#config_path = '../config/config_x.ini'   # will be deleted
#opt_type = 'train'                       # will be deleted

config = ConfigParser()
config.read(config_path)

root_dir = config.get('model', 'root_dir')
data_dir = config.get('data', 'data_dir')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# image,patientID,metaphase,label,side
# pID1.005-K_11_0_.png,pID1,005,11,0
if opt_type == 'train':
    img_labels = pd.read_csv('../data/train_data.csv') # '../data/train_data.csv'
elif opt_type == 'test':
    img_labels = pd.read_csv('../data/test.csv')

tr_image = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(224),
                    transforms.CenterCrop(224)
                    ])

l_all_images = []
arr_sum = np.array([0.0, 0.0, 0.0])
arr_mean = np.array([0.0, 0.0, 0.0])
arr_std = np.array([0.0, 0.0, 0.0])

psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

count=0
for img_path in tqdm(img_labels['image']):
    img_path = f'{data_dir}{opt_type}/singles/{img_path}'
    img = pil_loader(img_path)
    img = tr_image(img)
#     img = asarray(img)
    #print(f'shape: {img.shape} ... type: {type(img)}')
    psum    += img.sum(axis        = [1, 2])/224/224
    psum_sq += (img ** 2).sum(axis = [1, 2])/224/224
    #l_all_images.append(img)
    count+=1
    if count % 1000 == 0:
        print(f'psum: {psum} ... psum_sq: {psum_sq}')
        mean_images = psum / count
        total_var  = (psum_sq / count) - (mean_images ** 2)
        std_images  = torch.sqrt(total_var)
        print(f'mean_images: {mean_images} ... std_images: {std_images}')
        #break

mean_images = psum / count
total_var  = (psum_sq / count) - (mean_images ** 2)
std_images  = torch.sqrt(total_var)
        
#arr_all_images = np.array(l_all_images)

#print(f'shape of arr_all_images: {arr_all_images.shape}')
#print(f'..........arr_shape: {arr_all_images[0].shape}')
#mean_images = np.mean(arr_all_images, axis=(0,2,3))
#std_images = np.std(arr_all_images, axis=(0,2,3))
print(f'mean_images: {mean_images}')
print(f'std_images: {std_images}')
