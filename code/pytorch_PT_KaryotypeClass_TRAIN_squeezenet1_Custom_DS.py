### Google CoLab Configuration If Needed
# from google.colab import drive
# drive.mount('/content/drive')
# ## drive.mount("/content/drive", force_remount=True)

# ## !cp "/content/drive/My Drive/xy.py" "xy.py"
# %cd /content/drive/MyDrive/DL_Project/

# pip install humanfriendly

import train_model as tm

tm1 = tm.c_train_model()

### ******************* SET the Configuration Parameters ********************** ###
### Below are the defaults, uncomment and update them if needed
tm1.model_name = 'resnet18'  #"squeezenet1_0" 
# tm1.num_classes = 24           # Number of classes in the dataset
# tm1.batch_size = 64            # Batch size for training (change depending on how much memory you have)
tm1.num_epochs = 120           # Number of epochs to train for 30+70+50+50
# tm1.single_channel = False     # your images have 1 channel or 3 channels?
# ### Flag for feature extracting. When False, we finetune the whole model,
# ###   when True we only update the reshaped layer params ... usually set as False
# tm1.feature_extract = False
# tm1.use_pretrained = False          # if True, we ignore the pre-trained weights and train from scratch
# tm1.data_transform = 'data_transforms_2'      # 'data_transforms_2' or 'default'
# tm1.csv_name_pre = 'arkusai'

# tm1.partition_needed = False # creating directories once is enough, so, we need it for the first time only          
# ### Train/Val/Test directories are here with images in them
# tm1.data_dir = 'data/BioImLab_single_chromosomes_05.04.2021/'
# tm1.custom_dataset = True
# tm1.class_folders = False # whether each class has its own directory or all classes are in one single directory 'val'
# tm1.shuffle = True
# tm1.p_requires_grad = True 
# tm1.show_params = False
# tm1.use_trained_model = False
# tm1.pre_trained_file = 'models/resnet18_26epochs_0.7413.weights'

### tm1.dops1, tm1.datasets_dict, tm1.dataloaders_dict, tm1.optimizer_ft, tm1.criterion
### tm1.model_ft, tm1.hist, tm1.best_preds, tm1.best_labels, tm1.best_outputs, tm1.best_acc

tm1.run_all()
