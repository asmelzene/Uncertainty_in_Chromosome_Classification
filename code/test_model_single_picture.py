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

from PIL import Image
import os

import test_uncertainty_ReLU as tuc 
from humanfriendly import format_timespan
import pickle 
import datetime 

def main():
    start_date=datetime.datetime.now()
    print(f'start_date {start_date}')
    
    tm1 = test_model()

    #tm1.data_dir = 'data/BioImLab_single_chromosomes_05.04.2021/test/'
    #tm1.model_dir = 'models'
    tm1.model_list = {'squeezenet1'} #{'alexnet', 'densenet121', 'resnet18', 'squeezenet1', 'VGG11_bn'}
    tm1.model_meta_csv = 'models/model_meta_squeezenet1.csv' #'model_meta_densenet121.csv'  ## model_meta.csv
    tm1.img_dir = 'data/BioImLab_single_chromosomes_05.04.2021/test/'
    tm1.picture = '11/28 11b.bmp'
    #tm1.test_data_csv = 'data/BioImLab_single_chromosomes_05.04.2021/chromosome_data_test.csv'

    print(f'model running: {datetime.datetime.now()}')
    df_result_ensemble = tm1.run()

    pd.set_option('max_colwidth', -1)
    df_result_ensemble

    # tm1.df_result
    print(f'uncertainty running: {datetime.datetime.now()}')

    df_result_ensemble.to_pickle('results/df_result_ensemble2_new.pkl')

    with open('results/df_result_ensemble2_new.pkl', 'rb') as fp:
         df_result_ensemble = pickle.load(fp)

    df_result_ensemble

    tuc1 = tuc.test_uncertainty(df_result_ensemble, single_picture=True)

    tuc1.run()
    df_final = tuc1.df_uncertainty
    df_final_summary = tuc1.df_final_summary

    pd.set_option('max_colwidth', -1)
    pd.set_option('display.max_columns', None)
    print(df_final)

    print(df_final_summary)

    end_date=datetime.datetime.now()
    print(f'ALL done: {end_date}')

    t_delta = end_date - start_date
    format_timespan(t_delta.seconds)

class test_model:
    def __init__(self):
        self.data_dir = 'data/BioImLab_single_chromosomes_05.04.2021/test/' # not needed for MNIST_fashion
        self.model_dir = 'models'
        self.model_list = {'resnet18'} #{'alexnet', 'densenet121', 'resnet18', 'squeezenet1', 'VGG11_bn'}
        self.num_classes = 24
        self.batch_size = 64
        self.single_channel = False
        self.model_meta_csv = 'models/model_meta_resnet18.csv' #'model_meta_densenet121.csv'  ## model_meta.csv
        self.test_data_csv = 'data/BioImLab_single_chromosomes_05.04.2021/chromosome_data_test.csv'
        self.data_transform = 'default'
        self.class_folders = True # whether each class has its own directory or all classes are in one single directory 'test'
        self.img_dir = 'data/BioImLab_single_chromosomes_05.04.2021/test/'
        self.picture = '9/21 9b.bmp'
    
    def run(self, model_type = 'single', n_limit = None):
        df_model_meta = pd.read_csv(self.model_meta_csv)  
        df_image_list = pd.read_csv(self.test_data_csv)

        if self.data_transform == 'default':
            data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            data_transform = self.data_transform
              
        l_pictures = []
        #l_inputs = []
        l_outputs = []
        l_labels = []    
        l_model = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        counter = 0
        for i_m, m in tqdm(enumerate(df_model_meta.values)):
            count = 0
            model_name = m[0]
            model_path = Path(f'{self.model_dir}/{m[2]}')
            config = [self.data_dir, model_name, self.num_classes, self.batch_size, self.single_channel]
            
            pre_model_test = ppm.pt_model(config)
            model_test, input_size = pre_model_test.initialize()            
            model_test.load_state_dict(torch.load(model_path, map_location=device))
            model_test.eval()
            
            img_dir = self.img_dir
            picture = self.picture
            img_path = os.path.join(img_dir, picture)
            
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
            image = self.pil_loader(img_path)
            #image = np.expand_dims(image, 1)
            image = transform(image)
            ### ********* unsqueeze is needed since PyTorch expects a batch (64, 3, 224, 224) 
            # whereas we pass a single image (3, 224, 224)
            image = image.unsqueeze(0)    # Add new dimension at position 0, so >> (1, 3, 224, 224)
            
            #image = image[None, :, :, :]         # >>>> does the same job as unsqueeze
            #image = image.view(1, 3, 224, 224)   # >>>> does the same job as unsqueeze
            #image = image.reshape(1, 3, 224, 224)   # >>>> does the same job as unsqueeze
            try:
                label = picture.split('/')[0]
            except:
                label = ''
                
            image = image.to(device)
            output = model_test(image)

            # (picture, image, label_idx, label)
            l_pictures.extend([picture])                
            l_outputs.extend(output.detach().numpy())
            #l_labels.extend(labels.detach().numpy())
            l_labels.extend([label])
            l_model.extend([m[1]])
                
        self.l_pictures=l_pictures
        self.l_labels=l_labels
        self.l_model=l_model
        self.l_outputs=l_outputs
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
    
    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
if __name__ == "__main__":
    main()