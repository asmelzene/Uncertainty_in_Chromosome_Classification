#import os
#import numpy as np
import subprocess

#l_seed = [97, 81, 87, 51, 59, 23, 1, 3, 5, 7, 11, 15]
l_seed = [23, 1, 3, 5, 7, 11, 15]
#l_seed = [30, 50, 70]
# 3, 5, 7 didn't work well for squeezenet **************************
#l_seed = [11, 15]
# !! when we used numpy to generate random numbers, it gave an error in subprocess train_model even though it updated the conf file
# apparently, there is some conflict using other packages with subprocess in the same file... needs to be investigated later
for rand_seed in l_seed:
    #rand_seed = np.random.randint(200)
    #if rand_seed not in l_seed:
    #l_seed.append(rand_seed)
    #print(l_seed)
    print(f'rand_seed: {rand_seed} is in progress')
    # python ../config/update_config.py --config ../config/config_densenet121.ini --config_new ../config/config_densenet121.ini --section data --key seed --value 97
    #p1 = subprocess.Popen(['python', '../config/update_config.py', '--config', '../config/config_densenet121_TTA.ini', '--config_new', '../config/config_densenet121_TTA.ini', '--section', 'data', '--key', 'seed', '--value', str(rand_seed)])
    #p1 = subprocess.Popen(['python', '../config/update_config.py', '--config', '../config/config_densenet121.ini', '--config_new', '../config/config_densenet121.ini', '--section', 'data', '--key', 'seed', '--value', str(rand_seed)])
    #p1 = subprocess.Popen(['python', '../config/update_config.py', '--config', '../config/config_resnet18.ini', '--config_new', '../config/config_resnet18.ini', '--section', 'data', '--key', 'seed', '--value', str(rand_seed)])
    p1 = subprocess.Popen(['python', '../config/update_config.py', '--config', '../config/config_alexnet.ini', '--config_new', '../config/config_alexnet.ini', '--section', 'data', '--key', 'seed', '--value', str(rand_seed)])
    p1.wait()
    # python train_model.py --config ../config/config_densenet121.ini
    #p2 = subprocess.Popen(['python', 'train_model.py', '--config', '../config/config_densenet121.ini'])
    #p2 = subprocess.Popen(['python', 'train_model.py', '--config', '../config/config_resnet18.ini'])
    p2 = subprocess.Popen(['python', 'train_model.py', '--config', '../config/config_alexnet.ini'])
    p2.wait()
