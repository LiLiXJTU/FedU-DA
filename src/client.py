import gc
import pickle
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from local.fusiontrans import UFusionNet
from src.loss import DiceLoss, MultiClassDiceLoss, Focal_Loss,FocalLoss_Ori,Marginal_Loss
from src.utils import save_on_batch
from PIL import Image
logger = logging.getLogger(__name__)


# def dice_show(preds, masks, dataloader):
#     loader = dataloader
#     time_sum, loss_sum = 0, 0
#     dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
#     batch_size = preds.shape[0]
#     dices = []
#
#     for i, (sampled_batch, names) in enumerate(loader, 1):
#         # try:
#         #     loss_name = criterion._get_name()
#         # except AttributeError:
#         #     loss_name = criterion.__name__
#
#         # Take variable and put them to GPU
#         # images, masks = sampled_batch['image'], sampled_batch['label']
#         # images, masks = images.cuda(), masks.cuda()
#
#         # ====================================================
#         #             Compute loss
#         # ====================================================
#
#         # preds = model(images)
#         # out_loss, DICE_LOSS, BCE_LOSS = criterion(preds, masks.float())  # Loss
#
#         # if model.training:
#         #     optimizer.zero_grad()
#         #     out_loss.backward()
#         #     optimizer.step()
#
#         # print(masks.size())
#         # print(preds.size())
#
#         # train_iou = 0
#         # train_iou = iou_on_batch(masks, preds)
#         preds[preds >= 0.5] = 1
#         preds[preds < 0.5] = 0
#         # print("2",np.sum(tmp))
#         targets = masks.float()
#         targets[targets > 0] = 1
#         targets[targets <= 0] = 0
#         dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
#         train_dice = 1.0 - dice_loss(preds, targets)
#
#         # batch_time = time.time() - end
#         # # train_acc = acc_on_batch(masks,preds)
#         # if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
#         #     vis_path = config.visualize_path + str(epoch) + '/'
#         #     if not os.path.isdir(vis_path):
#         #         os.makedirs(vis_path)
#         #     save_on_batch(images, masks, preds, names, vis_path)
#         dices.append(train_dice)
#
#         # time_sum += len(images) * batch_time
#         # loss_sum += len(images) * out_loss
#         # iou_sum += len(images) * train_iou
#         # acc_sum += len(images) * train_acc
#         dice_sum += len(preds) * train_dice
#
#         if i == len(loader):
#             # average_loss = loss_sum / (config.batch_size * (i - 1) + len(images))
#             # average_time = time_sum / (config.batch_size * (i - 1) + len(images))
#             # train_iou_average = iou_sum / (config.batch_size * (i - 1) + len(images))
#             # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
#             train_dice_avg = dice_sum / (batch_size * (i - 1) + len(preds))
#         else:
#             # average_loss = loss_sum / (i * batch_size)
#             # average_time = time_sum / (i * batch_size)
#             # train_iou_average = iou_sum / (i * batch_size)
#             # train_acc_average = acc_sum / (i * config.batch_size)
#             train_dice_avg = dice_sum / (i * batch_size)
#     return train_dice_avg

def dice_show(preds, masks):
    # preds[preds >= 0.5] = 1
    # preds[preds < 0.5] = 0
    # # print("2",np.sum(tmp))
    # targets = masks.float()
    # targets[targets > 0] = 1
    # targets[targets <= 0] = 0
    # dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
    # train_dice = 1.0 - dice_loss(preds, targets)
    smooth = 1e-5
    intersection = (preds * masks).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + masks.sum() + smooth)
    return dice
class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, local_valdata, device,save_img_path):
        """Client object is initiated by the center server."""
        self.id = client_id
        #本地训练数据
        self.data = local_data
        self.valdata = local_valdata
        self.device = device
        self.__model = None
        self.save_img_path = save_img_path

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        #本地数据集
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.valdataloader = DataLoader(self.valdata, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        #self.criterion = WeightedDiceLoss
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    def client_update(self,idx):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)
        client_train_losses =[]
        optimizer = eval(self.optimizer)(self.model.parameters(), lr=2e-4,betas=(0.9, 0.999), weight_decay=5e-4)
        # optimizer =self.optimizer
        for e in range(self.local_epoch):
            for dataset, labelName in self.dataloader:
                #data, labels = data.float().to(self.device), labels.long().to(self.device)
                data = dataset['image'].float().to(self.device)
                labels = dataset['label'].float().to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                outputssoft = torch.softmax(outputs, dim=1)
                #outputs = torch.max(outputs, dim=1).values.unsqueeze(1)
                #outputs = torch.argmax(outputs,dim=1)
                #outputs = outputs.data.max(1)[1]
                # img_path = './train_img/'
                # save_on_batch(data, labels, outputssoft, labelName, img_path)
                # mDiceLoss, mDice = MultiClassDiceLoss()(outputssoft, labels)
                # floss = FocalLoss_Ori()(outputs,labels)
                # loss = mDiceLoss+floss
                l_ce,l_dice = Marginal_Loss()(outputssoft,labels,idx)
                print('l_dice',l_dice.item())
                loss = l_ce+l_dice
                print('loss', loss.item())
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                client_train_losses.append(loss.item())
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        client_avg_loss = np.average(client_train_losses)
        message = f"\t[Client {str(self.id).zfill(4)}] ...finished training!\
            \n\t=> Tain loss: {client_avg_loss:.4f}\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()
        return client_avg_loss,optimizer

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)
        client_id = self.id
        test_loss, dice_pred = 0, 0
        val_losses = []
        val_Dice = []
        with torch.no_grad():
            for dataset, labelName in self.valdataloader:
                #data, labels = data.float().to(self.device), labels.long().to(self.device)
                data = dataset['image'].float().to(self.device)
                labels = dataset['label'].float().to(self.device)
                outputs = self.model(data)
                outputssoft = torch.softmax(outputs,dim=1)
                mask = torch.argmax(outputs, dim=1)

                img_path = self.save_img_path + 'client_img/'+str(client_id)+'/'
                for i in range(labels.shape[0]):
                    labels_arr = labels[i][0].cpu().numpy()
                    mask_arr = mask[i].cpu().numpy()
                    # 定义颜色映射表
                    color_map = {
                        0: (0, 0, 0),  # 类别0为黑色
                        1: (221, 160, 221),  # 类别1为梅红色
                        2: (0, 128, 0),  # 类别2为深绿色
                        3: (128, 128, 0),  # 类别3为深黄色
                        4: (240, 255, 255),  # 类别4为深蓝色
                        5: (128, 0, 128),  # 类别5为深紫色
                        6: (0, 255, 255),  # 类别6为青色
                        7: (128, 128, 128),  # 类别7为灰色
                        8: (255, 0, 0),  # 类别8为红色
                        9: (0, 255, 0),  # 类别9为绿色
                        10: (255, 255, 0),  # 类别10为黄色
                        11: (0, 0, 255),  # 类别11为蓝色
                    }

                    # 将像素值映射成颜色
                    colors = np.zeros((*mask_arr.shape, 3), dtype=np.uint8)
                    for j, color in color_map.items():
                        colors[mask_arr == j, :] = color

                    # 将颜色数组保存为图片
                    # img = Image.new('RGB', (colors.shape[1], colors.shape[0]))
                    # data = colors.reshape(-1, colors.shape[2]).tolist()
                    # img.putdata(data)
                    # img.save(img_path + labelName[i][:-4] + ".png")
                    img = Image.fromarray(colors)
                    img.save(img_path + labelName[i][:-4] + ".png")


                    # 将像素值映射成颜色
                    gt_colors = np.zeros((*labels_arr.shape, 3), dtype=np.uint8)
                    for j, color in color_map.items():
                        gt_colors[labels_arr == j, :] = color

                    img = Image.fromarray(gt_colors)
                    img.save(img_path + labelName[i][:-4] + "_gt.png")
                #save_on_batch(data, labels, outputssoft, labelName, img_path)
                mDiceLoss, mDice = MultiClassDiceLoss()(outputssoft, labels)
                floss = FocalLoss_Ori()(outputs, labels)
                test_loss = mDiceLoss+floss

                val_losses.append(test_loss.item())
                val_Dice.append(mDice.item())
                # predicted = outputs.argmax(dim=1, keepdim=True)
                # correct += predicted.eq(labels.view_as(predicted)).sum().item()

                # dice_pred_t = dice_show(outputs, labels)
                # dice_pred += dice_pred_t

                #iou_pred += iou_pred_t
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = np.average(val_losses)
        #test_dice = dice_pred / len(self.valdataloader)
        test_dice = np.average(val_Dice)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> test_dice: {100. * test_dice:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()
        return test_loss, test_dice


# def show_image_with_dice(predict_save, labs):
#     # tmp_lbl = labs.type(torch.float32)
#     # tmp_3dunet = (predict_save).astype(np.float32)
#     # dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
#     # # dice_show = "%.3f" % (dice_pred)
#     # # iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
#     # return dice_pred
#     smooth = 1e-5
#     num = len(labs)
#     m1 = predict_save.view(num,-1)  # Flatten
#     m2 = labs.view(num, -1)  # Flatten
#     intersection = (m1 * m2).sum()
#     dice_pred = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
#     return dice_pred
#
# def vis_heatmap(model, input_img, labs):
#     model.eval()
#     img_size = 224
#     output = model(input_img.cuda())
#     pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
#     # predict_save = pred_class[0].cpu().data.numpy()
#     # predict_save = np.reshape(predict_save, (img_size, img_size))
#     dice_pred_tmp = show_image_with_dice(pred_class, labs)
#     return dice_pred_tmp

# def vis_heatmap(model, input_img, labs):
#     model.eval()
#     output = model(input_img.cuda())
#     #之前预测的Dice
#     # pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
#     # dice_pred_tmp = show_image_with_dice(pred_class, labs)
#     #不知道从哪里来的
#     # predict_save = pred_class[0].cpu().data.numpy()
#     # predict_save = np.reshape(predict_save, (img_size, img_size))
#     img_size = 224
#     #tensor(16,1,224,224)
#     pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
#     #ndarry(1,224,224)
#     predict_save = pred_class[0].cpu().data.numpy()
#     #ndarry(224,224)
#     predict_save = np.reshape(predict_save, (img_size, img_size))
#     dice_pred_tmp = show_image_with_dice(predict_save, labs)
#     return dice_pred_tmp
#
# def show_image_with_dice(predict_save, labs):
#
#     #tmp_lbl = (labs).astype(np.float32)
#     #tensor(16,224,224)
#     tmp_lbl = labs.type(torch.float32)
#     #tmp_lbl = labs
#     #ndarry(224,224)
#     tmp_3dunet = (predict_save).astype(np.float32)
#     #
#     dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
#     # dice_show = "%.3f" % (dice_pred)
#     return dice_pred



