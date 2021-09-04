from configparser import ConfigParser

#Get the configparser object
config_object = ConfigParser()

config_object['model'] = {
    'model_name': 'alexnet',
    'use_trained_model': False,
    'num_epochs': 100,
    'num_classes': 24,
    'batch_size': 64,
    'single_channel': False,
    'feature_extract': False,
    'p_requires_grad': True,
    'show_params': True,
    'root_dir': '/home/melih/arkusai/'
    'use_pretrained': False,
    'prev_model': '/home/melih/arkusai/models/resnet18_24.04.2021_00.25.51_arkusai_25epochs_0.7045.pth',
}

config_object['data'] = {
    'dataset_name': 'arkusai', 
    'data_dir': '/Datasets/arkusai/processed/',
    'custom_dataset': True,
    'data_transform': 'data_transforms_2',
    'shuffle': True,    
    'class_folders': False,
    'partition_needed': False,
    'index2class_csv': '/home/melih/arkusai/data/index_to_class.csv'
}

config_object['optimization'] = {
    'LearningRate': 1e-4,
    'WeightDecay': 1e-4,
    'Momentum': 0.9,
    'use_CLR': False,
    'base_LR': 0.0001,
    'max_LR': 0.1
}

config_object['train'] = {
    config_object['1'] = {
        'LearningRate': 1e-4,
        'WeightDecay': 1e-4,
        'Momentum': 0.9,
        'use_CLR': False,
        'base_LR': 0.0001,
        'max_LR': 0.1
    }
}

#Write the above sections to config.ini file
with open('config.ini', 'w') as conf:
    config_object.write(conf)
    
"""
from configparser import ConfigParser

#Read config.ini file
config = ConfigParser()
config.read('config.ini')

#Get some configs
conf_model, conf_data, conf_opt = config['model'], config['data'], config['optimization']
print('Model name: {}'.format(conf_model['model_name']))
print('use_trained_model: {}'.format(conf_model.getboolean('use_trained_model')))
print('num_epochs: {}'.format(conf_model.getint('num_epochs')))
print('custom_dataset: {}'.format(conf_data.getboolean('custom_dataset')))
print('LearningRate: {}'.format(conf_opt.getfloat('LearningRate')))
#print('use_trained_model: {}'.format(config.getboolean('model','use_trained_model')))

#Update the model_name
conf_model["model_name"] = "resnet18"

#Write changes back to file
with open('config.ini', 'w') as conf:
    config.write(conf)
"""