import argparse
from configparser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../config/config.ini', help='which config file to use')
parser.add_argument('--config_new', type=str, default='../config/config.ini')
parser.add_argument('--section', type=str, default='data')
parser.add_argument('--key', type=str, default='transform')
parser.add_argument('--value', type=str, default='data_transforms_3')

opt = parser.parse_args()

#Read config.ini file
config = ConfigParser()
config.read(opt.config)

conf_data = config[opt.section]
conf_data[opt.key] = opt.value
print('writing changes')
#Write changes back to file
with open(opt.config_new, 'w') as conf:
    config.write(conf)
