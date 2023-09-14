# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:14 下午
# @Author  : Haonan Wang
# @File    : Train_one_epoch.py
# @Software: PyCharm
import torch.optim
import os
import time
import Config as config
import warnings

from local.localutils import save_on_batch
from src.loss import DiceLoss,MultiClassDiceLoss

warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = False

def print_summary(epoch, i, nb_batch, loss,
                  average_loss, average_time,
                  dice, average_dice, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    # string += 'IoU:{:.3f} '.format(iou)
    # string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, logger):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0

    dices = []
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = 'DiceLoss'
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks = sampled_batch['image'], sampled_batch['label']
        images, masks = images.cuda(), masks.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        preds = model(images)
        outputssoft = torch.softmax(preds, dim=1)
        # out_loss = DiceLoss(2)(outputssoft, masks.unsqueeze(1))
        #out_loss = WeightedDiceLoss()(preds, masks.unsqueeze(1))
        out_loss = MultiClassDiceLoss()(outputssoft, masks)
        # out_loss = MultiClassDiceLoss()(preds, masks)
        # out_loss = danDiceLoss(preds, masks.unsqueeze(1))
        # out_loss = cal_dice(outputssoft, masks.unsqueeze(1),2)
        # out_loss,DICE_LOSS,BCE_LOSS = criterion(preds, masks.float())  # Loss


        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        # print(masks.size())
        # print(preds.size())


        # train_iou = 0
        # train_iou = iou_on_batch(masks,preds)
        # train_dice = criterion._show_dice(preds, masks.float())
        train_dice = 1 - out_loss
        batch_time = time.time() - end
        # train_acc = acc_on_batch(masks,preds)
        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        # iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss,
                          average_loss, average_time,
                          train_dice, train_dice_avg, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)
            # writer.add_scalar(logging_mode + 'DICE_LOSS', DICE_LOSS, step)
            # # writer.add_scalar(logging_mode + 'BCE_LOSS', BCE_LOSS, step)
            # writer.add_scalar(logging_mode + 'DICE_LOSS_epoch', DICE_LOSS, epoch)
            # writer.add_scalar(logging_mode + 'BCE_LOSS_epoch', BCE_LOSS, epoch)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()
    # if epoch + 1 > 10: # Plateau
    #     if lr_scheduler is not None:
    #         lr_scheduler.step(train_dice_avg)
    return average_loss, train_dice_avg

