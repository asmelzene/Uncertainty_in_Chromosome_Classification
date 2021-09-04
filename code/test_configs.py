#import os
import subprocess

#os.system("python myOtherScript.py arg1 arg2 arg3")  
l_transform = ['data_transforms_31', 'data_transforms_32', 'data_transforms_33']

for tr in l_transform:
  p1 = subprocess.Popen(['python', '../config/update_config.py', '--value', tr])
  p1.wait()
  p2 = subprocess.Popen(['python', 'train_model.py'])
  #p2 = subprocess.Popen(['python', 'train_model.py', '--config', '../config/config_squeezenet1_0.ini'])
  p2.wait()

#for tr in l_transform:
    #os.system(f'python ../config/update_config.py --value {tr}')
    #os.system('python train_model.py')

