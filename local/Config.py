# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:44 下午
# @Author  : Haonan Wang
# @File    : Config.py
# @Software: PyCharm
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 500  #666 500
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True # whether use cosineLR or not
n_channels = 3
n_labels = 2
epochs = 50
img_size = 224
print_frequency = 20
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False
task_name = 'CTseg' # GlaS MoNuSeg luna16 lung LungSeg
# task_name = 'GlaS'
learning_rate = 2e-4
trans_lr = 0
batch_size = 4

#model_name = 'UCTransNet3Plus_follow  '
#model_name = 'SAUnetPP'
#model_name = 'UCTransNet_pretrain  '
model_name = 'UFusionNet'

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'  #Test_Folder  Val_Folder
test_dataset = 'C:/code/fedsemi-l/data/UESTC70new/Test_Folder/'
session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4  960 896
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 32 # base channel of U-Net
    config.n_classes = 1
    config.hidden_size = 196
    return config




# used in testing phase, copy the session name in training phase
test_session = "Test_session_06.11_16h45"