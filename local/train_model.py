# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @Author  : Haonan Wang
# @File    : train.py
# @Software: PyCharm
import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
import Config
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
import yaml

from local.UfNet import UCTransNet
from local.fusiontrans import UFusionNet
from src.LoadData import LoadDatasets
from src.models import UNet


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)

    #torch.save({'model': lily.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i, 'loss_array': loss_list, 'acc_array': acc_list}, '../model_dict/resnet50model_with_pretrain.dict')


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    with open('C:/code/fedsemi-l/config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]
    UESTC, CTseg, MosMed, medseg, test_dataset, val_UESTC, val_CTseg, val_MosMed, val_medseg = LoadDatasets()
    train_dataset=UESTC
    val_dataset=val_UESTC
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=0,
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=0,
                            pin_memory=True)

    lr = config.learning_rate
    logger.info(model_type)
  # ---------------------------------------------------------------------------------------------
    if model_type == 'UCTransNet':
        model = UCTransNet(**model_config)

    elif model_type == 'UCTransNet3Plus_follow':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand+ ratio: {}'.format(config_vit.expand_ratio))
        model = UNet(**model_config)
        pretrained_UNet_model_path = "./lung/UCTransNet3Plus/Test_session_05.19_10h05/models/best_model-UCTransNet3Plus.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict,strict=False)
        logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    for name, p in model.named_parameters():
        if name.startswith('mtc'):
            p.requires_grad = False
        if name.startswith('inc'):
            p.requires_grad = False

    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])
    criterion = fed_config["criterion"]
    #criterion = Dice_TopK_BD(dice_weight=0.02,TopK_weight=0.4,BD_weight=0.58)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4,betas=(0.9, 0.999), weight_decay=5e-4)
    # Choose optimize
    # optimizer = torch.optim.Adam([
    #     {"params":model.mtc.parameters(),"lr":Config.trans_lr},
    #     {"params": model.inc.parameters(), "lr": Config.trans_lr}
    # ],lr=lr)
    # if config.cosineLR is True:
    #     lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-9)
    #     #lr_scheduler = MultiStepLR(optimizer,milestones=[100],gamma=0.5)
    # else:
    lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        # img, _ = train_dataset.__getitem__(0)
        # img = img['image'].cuda()
        # img = img.unsqueeze(0)
        # writer.add_graph(model, img)
        # writer.close()
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, logger)
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                 optimizer, writer, epoch, lr_scheduler, logger)

        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 2:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 # 'dice_loss':dice_loss,
                                 # 'BCE_loss':BCE_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
