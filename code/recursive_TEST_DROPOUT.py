import subprocess

#python test_model_DROPOUT.py --config ../config/config_densenet121_DROPOUT.ini
'''
print(f'Alexnet-TEST is in progress')
p1 = subprocess.Popen(['python', 'test_model_DROPOUT.py', '--config', '../config/config_alexnet_DROPOUT.ini'])
p1.wait()
print(f'Resnet18-TEST is in progress')
p2 = subprocess.Popen(['python', 'test_model_DROPOUT.py', '--config', '../config/config_resnet18_DROPOUT.ini'])
p2.wait()
'''
print(f'Densenet121-TEST is in progress')
p3 = subprocess.Popen(['python', 'test_model_DROPOUT.py', '--config', '../config/config_densenet121_DROPOUT.ini'])
p3.wait()
'''
print(f'Squeezenet1_0-TEST is in progress')
p4 = subprocess.Popen(['python', 'test_model_DROPOUT.py', '--config', '../config/config_squeezenet1_0_DROPOUT.ini'])
p4.wait()
'''
