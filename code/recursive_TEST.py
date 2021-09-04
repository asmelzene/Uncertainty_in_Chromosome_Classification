import subprocess

'''
print(f'Alexnet-TEST is in progress')
#p1 = subprocess.Popen(['python', 'test_model.py', '--config', '../config/config_alexnet.ini'])
p1 = subprocess.Popen(['python', 'test_model.py', '--config', '../config/config_alexnet_SNAP.ini'])
p1.wait()
print(f'Resnet18-TEST is in progress')
#p2 = subprocess.Popen(['python', 'test_model.py', '--config', '../config/config_resnet18.ini'])
p2 = subprocess.Popen(['python', 'test_model.py', '--config', '../config/config_resnet18_SNAP.ini'])
p2.wait()
'''
print(f'Densenet121-TEST is in progress')
#python test_model.py --config ../config/config_densenet121_snap.ini
p3 = subprocess.Popen(['python', 'test_model.py', '--config', '../config/config_densenet121_SNAP.ini'])
#p3 = subprocess.Popen(['python', 'test_model.py', '--config', '../config/config_densenet121_TTA_Ens.ini'])
p3.wait()
'''
print(f'Squeezenet1_0-TEST is in progress')
#p4 = subprocess.Popen(['python', 'test_model.py', '--config', '../config/config_squeezenet1_0.ini'])
p4 = subprocess.Popen(['python', 'test_model.py', '--config', '../config/config_squeezenet1_0_SNAP.ini'])
p4.wait()
'''
