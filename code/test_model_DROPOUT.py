# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/DL_Project/

import torch
import torch.nn as nn
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
        
        #torch.manual_seed(self.config.getint('data', 'seed'))
        #np.random.seed(self.config.getint('data', 'seed'))
        #torch.use_deterministic_algorithms(True)
        #torch.backends.cudnn.benchmark = False
        #torch.set_deterministic(True)
        #torch.backends.cudnn.deterministic = True
        
        dops1 = do.dops(config)
        self.dops1 = dops1   
        self.logger = dops1.get_logger('test')
        self.logger.info(f'config_path: {config_path}')
        
#         print(f'config_path: {config_path}')
#         logging.info(f'config_path: {config_path}')

        self.root_dir = config.get('model', 'root_dir')
        self.drop_rate = config.getfloat('model', 'drop_rate')
        self.n_forward = config.getint('model', 'n_forward')
        self.data_dir = self.data_dir = config.get('data', 'data_dir')
        #self.model_dir = '/home/melih/arkusai/models'
        #{'alexnet', 'densenet121', 'resnet18', 'squeezenet1', 'VGG11_bn'}
        self.model_list = {config.get('model', 'name')} 
        self.num_classes = config.getint('model', 'num_classes')
        self.batch_size = config.getint('model', 'batch_size')
        self.single_channel = config.getboolean('model', 'single_channel')
        self.model_meta_csv = config.get('test', 'model_meta_csv') #'model_meta_densenet121.csv'  ## model_meta.csv
        self.test_data_csv = f'{self.root_dir}data/test_data.csv'
        #self.df_result_ensemble = config.get('test', 'df_result_ensemble')
        self.data_transform = self.data_transform = config.get('data', 'transform')
        self.class_folders = True # whether each class has its own directory or all classes are in one single directory 'test' 

    #def enable_dropout_Densenet(self, model, drop_rate = 0.1):
    def enable_dropout_Densenet(self, model):
        for m in model.modules():
            if m.__class__.__name__.startswith('_DenseBlock'):
                m.drop_rate = self.drop_rate
                m.train()
            elif m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    def run(self, model_type = 'single', n_limit = None):
        df_model_meta = pd.read_csv(f'{self.root_dir}models/{self.model_meta_csv}')

        if self.drop_rate > 0 and self.drop_rate < 1:
            model_name_dropout = df_model_meta.loc[0, 'model_name']
            for i in range(1, self.n_forward):
                df_model_meta.loc[i] = [model_name_dropout, f'{model_name_dropout}_{i}', df_model_meta.loc[0, 'model_path']]

        df_image_list = pd.read_csv(self.test_data_csv)

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
        # since we need all the output values, using GPU for that is exceeding the memory. that's why, we use CPU here
        #device = 'cpu'
        counter = 0
        for i_m, m in tqdm(enumerate(df_model_meta.values)):
            count = 0
            model_name = m[0]
            model_path = Path(f'{self.root_dir}models/{m[2]}')
            self.logger.info(f'model_path: {model_path}')
            
            pre_model_test = ppm.pt_model(self.config)
            model_test, input_size = pre_model_test.initialize()        
            
            self.dops1.pick_data_transforms(self.data_transform, pre_model_test.input_size)
            model_test.load_state_dict(torch.load(model_path, map_location=device))

            if device == 'cpu':
                model_test.cpu()

            model_test.eval()
            if self.drop_rate > 0 and self.drop_rate < 1:
                print(f'model: {m[1]}')
                #self.enable_dropout_Densenet(model_test, drop_rate = self.drop_rate)
                model_test.apply(self.enable_dropout_Densenet)
                
                if model_name == 'squeezenet1_0':
                    #final_conv = nn.Conv2d(512, 24, kernel_size=1)
                    #classifier_updated = nn.Sequential(
                    #    nn.Dropout(p=0.1*i_m),
                    #    final_conv,
                    #    nn.ReLU(inplace=True),
                    #    nn.AdaptiveAvgPool2d((1, 1))
                    #)
                    #model_test.classifier = classifier_updated.cuda()
                    #model_test.classifier[0] = nn.Dropout(p=0.1*i_m) 
                    model_test.classifier[0] = nn.Dropout(p=0.6)

            #if device == 'cpu':
            #    model_test.cpu()
                        
            test_dataset = cds.MyImageDataset(self.config, self.dops1, t_data_set = 'test')
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
#             test_dataset = cds.MyImageDataset(annotations_file = self.test_data_csv, img_dir = self.data_dir, 
#                              transform=data_transform, class_folders = self.class_folders)
#             test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)

            # (picture, image, label_idx, label)
            for pictures, inputs, _, labels in tqdm(test_dataloader):        
                inputs = inputs.to(device)
                #labels = labels.to(device)
                #pictures = pictures.to(device)
                #for l in labels.numpy():
                    #labels_real.append(all_real_labels[l])
                    #self.labels_real.extend(self.all_real_labels[l])
                
                outputs = model_test(inputs)
                #outputs = outputs.to(device)
                outputs = outputs.to('cpu')
                outputs = outputs.detach().numpy()

                l_pictures.extend(pictures)    
#                 if device == 'cpu':
#                     l_outputs.extend(outputs.detach().numpy())
#                 else:
#                     l_outputs.extend(outputs)
                
                l_outputs.extend(outputs)
                #l_labels.extend(labels.detach().numpy())
                l_labels.extend(labels)

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
    
