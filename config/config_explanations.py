[model]
name = alexnet    		# Models to choose from [resnet18, alexnet, VGG11_bn, squeezenet1_0, densenet121, inception_v3]
root_dir = /home/melih/arkusai/  # where we keep all the files/folders related to a project
#root_dir = ../         # for MAC-local PC
use_pretrained = False 	# if True, we ignore the pre-trained weights and train from scratch
num_epochs = 100    	# Number of epochs to train for: how many times we will go over (train) the entire data
num_classes = 24  		# Number of classes in the dataset, e.g. 24
batch_size = 64			# Batch size for training (use small values if memory fails), e.g. 64, 32, 16, .. 
drop_rate = -0.5        # Usually 0.5 The default interpretation of the dropout hyperparameter is the probability of training a given node in a layer, where 1.0 means no dropout, and 0.0 means no outputs from the layer. A good value for dropout in a hidden layer is between 0.5 and 0.8. Input layers use a larger dropout rate, such as of 0.8
n_forward = 50          # DROPOUT: how many times do you want to do forward pass (# of models to be created with random dropouts)
single_channel = False	# your images have 1 channel or 3 channels?
feature_extract = False # Flag for feature extracting. When False, we finetune the whole model,
        				# when True we only update the reshaped layer params ... usually set as False
p_requires_grad = True  # True, if training
show_params = False
use_prev_model = False  # when we are training, we regularly save model-weights in case we want to continue from 
						# where we left off. 
						# e.g. #self.pre_trained_file = '/home/melih/arkusai/models/resnet18_11epochs_0.7924.weights'
prev_model = /home/melih/arkusai/models/resnet18_24.04.2021_00.25.51_arkusai_25epochs_0.7045.pth
#prev_model = ../models/alexnet_02.05.2021_03.57.43_arkusai_80epochs_0.5985.pth

[data]
dataset_name = arkusai 					# on which dataset we work on
data_dir = /Datasets/arkusai/processed/ 	# Train/Val/Test directories are here with images in them
											# e.g. /Datasets/arkusai/processed/train/singles/
#data_dir = ../data/
val_dir = train    							# we will use some part of train or test as val e.g. train
custom_dataset = True						# needed to have control on what we want from data
transform = data_transforms_best   	# data transform for the data loader: e.g. 'data_transforms_best'
									        # e.g. ToTensor(), Resize(256), CenterCrop(224), etc.
transform_type = regular    # regular: we use regular torchvision.transforms
                            # albumentations: we use another package called albumentations for augmentation & transforms
shuffle = True              # shuffle the dataset, passed to DataLoader
class_folders = False		# whether each class has its own directory or all classes are in one single directory 'val' ... False
							# e.g. False > all classes are in > /Datasets/arkusai/processed/train/singles/
							# 	   True > class 1 is in > /Datasets/arkusai/processed/train/singles/1/
partition_needed = False 	# it is needed only if we received the raw data and preprocess it
							# creating directories once is enough, so, we need it for the first time only  
							# so, if we have files and folders created already  e.g. False
index2class_csv = /home/melih/arkusai/data/index_to_class.csv # index_to_class conversion file
#index2class_csv = ../data/index_to_class.csv
input_type = int_str        # int_str > 0,1,.....,22,23 == 1,2,.....,x,y
                            # int     > 0,1,.....,22,23 == 1,2,.....,23,24
                            # auto    > let dataloader decide it automatically, it will not be ordered
                            #         > 0,1,.....,22,23 == 1,11,12,...,23,..,8,9
seed = 59                   # seed for numpy and torch for random operations

[optimization]
optimizer = Adam            # if Adam is in use, CLR will be ignored. if SGD, then CLR flag will be checked (use_clr)
learningrate = 0.0001       # if simple SGD
weightdecay = 0.0001        # if simple SGD
momentum = 0.9              # if simple SGD
use_clr = False             # if TRUE, CLR related parameters will be used
cycle_mode = triangular     # CLR mode if CLR is in use
base_lr = 0.00001           # for CLR
max_lr = 0.1                # for CLR

[threshold]
acc_save = 0.74             # save each weights which are better than the previous one if acc>acc_save (VALIDATION)
acc_diff_save = 0.1         # save weights if the new weights are 0.1 better than the previous one, even if acc<acc_save,
                            # save the weights.
acc_stop = 0.99             # if TRAIN_ACC==0.99, stop training...assuming that the model will learn very little after this point
loss_stop = 0.01            # if VAL_LOSS==0.01, stop training...same reasining above

[test]
model_meta_csv = ../models/model_meta_squeezenet1_0.csv  # which model weights should be used for testing
df_result_ensemble = ../results/resnet18_29.04.2021_00.46.49_df_result_ensemble.pkl # which df_result_ensemble file will be used for uncertainty calculations
t_data_set = test           # val_test: if you want to see the results in validation set, more investigation in the data, since we don't save dataFrame for validation durin our training
                            # test: regular test related computations on test data
                            # while training data using train_model.py, we don't read this config, we hardcode the train and val
                            # so this separation is only for applying test calculations on test data or validation data
                            # in other words, it is considered with test_model.py but not train_model.py
n_TTA = 100                 # how many augmented images will be used in the test time for Test Time Augmentation (TTA)
                            # number of augmentations to be tested for each picture in TTA

[log]
level = info                # log level to log
