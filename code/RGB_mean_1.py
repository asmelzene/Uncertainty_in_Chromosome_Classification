import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import image
import data_ops as do
import custom_dataset as cds
import argparse
from configparser import ConfigParser
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../config/config.ini', help='which config file to use')
parser.add_argument('--opt', type=str, default='train')

opt = parser.parse_args()
config_path = opt.config
opt_type = opt.opt

config = ConfigParser()
config.read(config_path)

root_dir = config.get('model', 'root_dir')
data_dir = config.get('data', 'data_dir')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img = np.asarray(img)
        return img

# image,patientID,metaphase,label,side
# pID1.005-K_11_0_.png,pID1,005,11,0
if opt_type == 'train':
    img_labels = pd.read_csv('../data/train_data.csv') # '../data/train_data.csv'
elif opt_type == 'test':
    img_labels = pd.read_csv('../data/test.csv')

count=0
l_all_images = []
for img_path in tqdm(img_labels['image']):
    img_path = f'{data_dir}{opt_type}/singles/{img_path}'
    img = pil_loader(img_path)
    #img = image.imread(img_path)
    print(f'img_shape: {img.shape} ... img_type: {type(img)}')
    l_all_images.append(img)
    count += 1
    if count == 5:
        break

arr_all_images = np.array(l_all_images)
print(f'shape of arr_all_images: {arr_all_images.shape}')
print(f'..........{arr_all_images[0].shape}')
mean_images = np.mean(arr_all_images, axis=(0,1,2))
std_images = np.std(arr_all_images, axis(0,1,2))
print(f'mean_images: {mean_images}')
print(f'std_images: {std_images}')

#return arr_all_images
