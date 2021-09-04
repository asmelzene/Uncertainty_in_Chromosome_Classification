# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/DL_Project/

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_pretrained_models as ppm

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
from configparser import ConfigParser
import custom_dataset as cds
import data_ops as do
from PIL import Image
from scipy.special import softmax
from datetime import datetime
#[resnet, alexnet, vgg, squeezenet, densenet, inception]

# import logging

def main(config_path):        
    tm1 = test_model(config_path)
#     logging.basicConfig(filename='../logs/example.log', level=logging.INFO) # Creates log file    

    for section in tm1.config.sections():
        tm1.logger.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> section: {section}')
        for (key, val) in tm1.config.items(section):
            tm1.logger.info(f'{key}: {val}')            
    
    tm1.model_list = tm1.model_list         #{'alexnet', 'densenet121', 'resnet18', 'squeezenet1', 'VGG11_bn'}
    tm1.model_meta_csv = tm1.model_meta_csv #'model_meta_densenet121.csv'  ## model_meta.csv
    
    df_result_ensemble = tm1.run()
    
    f_name = tm1.dops1.generate_file_name(f'{tm1.root_dir}results', next(iter(tm1.model_list)), 'df_result_ensemble.pkl')
    df_result_ensemble.to_pickle(f_name)
    print(f'df_result_ensemble is saved in {f_name}')

class test_model:
    def __init__(self, config_path):
        config = ConfigParser()
        #config.read('../config/config.ini')
        config.read(config_path)
        self.config = config
        
        torch.manual_seed(self.config.getint('data', 'seed'))
        np.random.seed(self.config.getint('data', 'seed'))
        
        dops1 = do.dops(config)
        self.dops1 = dops1   
        self.logger = dops1.get_logger('test')
        self.logger.info(f'config_path: {config_path}')
        
#         print(f'config_path: {config_path}')
#         logging.info(f'config_path: {config_path}')

        self.root_dir = config.get('model', 'root_dir')
        self.data_dir = self.data_dir = config.get('data', 'data_dir')
        #self.model_dir = '/home/melih/arkusai/models'
        #{'alexnet', 'densenet121', 'resnet18', 'squeezenet1', 'VGG11_bn'}
        self.model_list = {config.get('model', 'name')} 
        self.num_classes = config.getint('model', 'num_classes')
        self.batch_size = config.getint('model', 'batch_size')
        self.shuffle = config.getboolean('data', 'shuffle')
        self.t_data_set = config.get('test', 't_data_set')          # 'test' or 'val_test'
        self.transform_type = config.get('data', 'transform_type')
        self.single_channel = config.getboolean('model', 'single_channel')
        self.model_meta_csv = config.get('test', 'model_meta_csv') #'model_meta_densenet121.csv'  ## model_meta.csv
        if self.t_data_set == 'test':
            self.test_data_csv = f'{self.root_dir}data/test_data.csv'
        elif self.t_data_set == 'val_test':
            self.test_data_csv = f'{self.root_dir}data/val_data.csv'
        self.n_TTA = config.getint('test', 'n_TTA')
        #self.df_result_ensemble = config.get('test', 'df_result_ensemble')
        self.data_transform = config.get('data', 'transform')
        self.class_folders = True # whether each class has its own directory or all classes are in one single directory 'test' 

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    '''
    index_to_class = pd.read_csv('../data/index_to_class.csv')

    def idx2class(self, idx):
        return index_to_class[index_to_class['index']==idx]['class'][idx]
    '''
    def run(self, model_type = 'single', n_limit = None):
        df_model_meta = pd.read_csv(f'{self.root_dir}models/{self.model_meta_csv}')
#         df_image_list = pd.read_csv(self.test_data_csv)

#         if self.data_transform == 'default':
#             data_transform = transforms.Compose([
#                     transforms.ToTensor(),
#                     transforms.Resize(256),
#                     transforms.CenterCrop(224),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ])
#         else:
#             data_transform = self.data_transform
              
        l_pictures = []
        #l_inputs = []
        l_outputs = []
        l_labels = []    
        l_model = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        counter = 0
        # df_model_meta contains models (weights) to be used for the ensemble
        for i_m, m in tqdm(enumerate(df_model_meta.values)):
            count = 0
            model_name = m[0]
            model_path = Path(f'{self.root_dir}models/{m[2]}')
            self.logger.info(f'model_path: {model_path}')
            
            pre_model_test = ppm.pt_model(self.config)
            model_test, input_size = pre_model_test.initialize()        
            
            self.dops1.pick_data_transforms(self.data_transform, pre_model_test.input_size)
            model_test.load_state_dict(torch.load(model_path, map_location=device))
            model_test.eval()
            
            if device == 'cpu':
                model_test.cpu()
                        
            test_dataset = cds.MyImageDataset(self.config, self.dops1, t_data_set = self.t_data_set)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

            # (picture, image, label_idx, label)
            aug_tst = 0 
            n_TTA = self.n_TTA
            dir_test_data = f'{self.data_dir}test/singles/'
            transform = self.dops1.data_transforms['val']

            if self.t_data_set == 'val_test':
                dir_test_data = f'{self.data_dir}train/singles/'
            
            for pictures, inputs, _, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                
                if n_TTA > 1:                           # means we will use TTA
                    for i in tqdm(range(len(labels))):
                        #print(f'1: {datetime.now()}')
                        np_output_single = np.zeros([1, 24])
                        img_path = f'{dir_test_data}{pictures[i]}'
                        image = self.pil_loader(img_path)
                        #print(f'2: {datetime.now()}')
                        if self.transform_type == 'albumentations':
                            image_arr = np.asarray(image)
                        #print(f'3: {datetime.now()}')
                        for j in range(n_TTA):                            
                            if self.transform_type == 'albumentations':
                                #print(f'4: {datetime.now()}')
                                single_input = transform(image=image_arr)
                                single_input = single_input['image']
                                single_input = single_input.to(device)
                            else:
                                single_input = transform(image)

                            #print(f'5: {datetime.now()}')
                            output_single = model_test(single_input.reshape([1, 3, 224, 224]))
                            output_single = output_single.to('cpu')
                            output_single = output_single.detach().numpy()
                            #output_single = softmax(output_single)
                            np_output_single += output_single

                        #print(f'6: {datetime.now()}')
                        l_pictures.append(pictures[i])
                        l_outputs.extend(np_output_single/n_TTA)                        
                        l_labels.append(labels[i])
                else:
                    outputs = model_test(inputs)
                    outputs = outputs.to('cpu')
                    outputs = outputs.detach().numpy()

                    l_pictures.extend(pictures)

                    l_outputs.extend(outputs)
                    l_labels.extend(labels)
            
            # which model is used for the above calculations:
            l_model.extend(list(np.full(int(len(l_labels)/(i_m + 1)), m[1])))

        df_result = pd.DataFrame({'picture': l_pictures, 'label': l_labels, 'model': l_model, 'outputs': l_outputs})

        ### >>>> Here we will MERGE the OUTPUTS OF the ensemble MODELS, merge the results in 1 row from 5 rows
        ### assuming that we have 5 models in the ensemble, before the merge we have 5 rows for 1 picture, then 1 row per picture
        pic_list = set(df_result['picture'])
        # model_list = set(df_result['picture'])
        #model_list = {'alexnet', 'densenet121', 'resnet18', 'squeezenet1', 'VGG11_bn'}
        model_list = self.model_list
        counter = 0
        df_result_ensemble = pd.DataFrame({'picture': [], 'model': [], 'label': [], 'outputs_all': []})
        for pic in tqdm(pic_list):
            if model_type == 'single':
                # e.g. 5 from resnet18
                tmp_model = list(model_list)[0]
                tmp_label = list(df_result[df_result['picture'] == pic]['label'])[0]
                #tmp_outputs = df_result[df_result['picture'] == pic]['outputs'].mean()
                tmp_outputs_all = np.array(df_result[df_result['picture'] == pic]['outputs'])

                df_result_ensemble.loc[counter] = [pic, tmp_model, tmp_label, tmp_outputs_all]
                #df_result_ensemble.loc[counter] = [pic, tmp_model, tmp_label, tmp_outputs, tmp_outputs_all]

                counter +=1
            elif model_type == 'mix-single':
                # e.g. 1 from all {'alexnet', 'densenet121', 'resnet18', 'squeezenet1', 'VGG11_bn'}
                tmp_model = 'mix'
                tmp_label = list(df_result[df_result['picture'] == pic]['label'])[0]
                #tmp_outputs = df_result[df_result['picture'] == pic]['outputs'].mean()
                tmp_outputs_all = np.array(df_result[df_result['picture'] == pic]['outputs'])

                df_result_ensemble.loc[counter] = [pic, tmp_model, tmp_label, tmp_outputs_all]
                #df_result_ensemble.loc[counter] = [pic, tmp_model, tmp_label, tmp_outputs, tmp_outputs_all]

                counter +=1
            elif model_type == 'mix-multi':  
                # this grouping by model is required only when you apply multiple different model-groups
                # e.g. 5 resnet18, 5 densenet
                for model in model_list:
                    tmp_model = model
                    tmp_label = list(df_result[(df_result['picture'] == pic) & 
                              (df_result['model'].str.contains(model, na=False))]['label'])[0]
                    #tmp_outputs = df_result[(df_result['picture'] == pic) & 
                     #         (df_result['model'].str.contains(model, na=False))]['outputs'].mean()
                    tmp_outputs_all = np.array(df_result[(df_result['picture'] == pic) & 
                              (df_result['model'].str.contains(model, na=False))]['outputs'])

                    df_result_ensemble.loc[counter] = [pic, tmp_model, tmp_label, tmp_outputs_all]

                    counter +=1
        
        self.df_result_ensemble = df_result_ensemble
        
        return df_result_ensemble

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/config.ini', help='which config file to use')
    
    opt = parser.parse_args()
    config_path = opt.config
    
    main(config_path)
    
