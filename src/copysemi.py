from random import random

from torch.utils.data import DataLoader, Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
import gc
import pickle
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.loss import WeightedDiceLoss,DiceLoss
from src.utils import save_on_batch

logger = logging.getLogger(__name__)

def get_current_consistency_weight(epoch):
    return 0.1 * sigmoid_rampup(epoch, 200.0)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # multi = ema_param.data * alpha
        # add_result = multi + param.data
        #教师模型
        #param是学生模型在第t次训练迭代中的参数
        ema_param.data = alpha * ema_param.data + (1-alpha) * param.data
        # ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        # print(add_result)
def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def dice_show(preds, masks):
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    # print("2",np.sum(tmp))
    targets = masks.float()
    targets[targets > 0] = 1
    targets[targets <= 0] = 0
    dice_loss = DiceLoss(n_classes=2)
    #dice_loss = WeightedDiceLoss()
    train_dice = 1.0 - dice_loss(preds, targets.unsqueeze(1))
    return train_dice

class UnsupervisedClient(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """

    def __init__(self, client_id, local_data, local_valdata, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        # 本地训练数据
        self.data = local_data
        self.valdata = local_valdata
        self.device = device
        self.__model = None

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
        # 本地数据集
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.valdataloader = DataLoader(self.valdata, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        # self.criterion = WeightedDiceLoss
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    #iter_num = 0
    def client_update(self):
        """Update local model using local dataset."""
        self.batch_size = 16
        self.num_classes = 2
        self.ema_decay = 0.99
        self.seed = 500
        torch.manual_seed(self.seed)
        #random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.model.train()
        self.model.to(self.device)
        # self.ema_model = self.model
        # self.ema_model.load_state_dict(self.model.state_dict())
        def create_model(ema=True):
            # Network definition
            ema_model = self.model
            if ema:
                for param in ema_model.parameters():
                    param.detach_()
            return ema_model
        self.ema_model = create_model(ema=True)
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        batch_loss = []
        iter_loss =[]
        iter_num = 0
        for iter in range(self.local_epoch):
            for dataset, labelName in self.dataloader:
                # data, labels = data.float().to(self.device), labels.long().to(self.device)
                data = dataset['image'].float().to(self.device)
                #labels = dataset['label'].float().to(self.device)
                #选取最小的4个
                unlabeled_volume_batch = data[4:]
                inputs = data
                noise = torch.clamp(torch.randn_like(
                    unlabeled_volume_batch) * 0.1, -0.2, 0.2)
                ema_inputs = unlabeled_volume_batch + noise
                outputs = self.model(inputs)

                #outputs_soft = torch.softmax(outputs, dim=1)
                with torch.no_grad():
                    # 无监督学习中的输出
                    ema_output = self.ema_model(ema_inputs)
                T = 8
                # (12,3,224,224)
                _, _, w, h = unlabeled_volume_batch.shape
                # (24,3,224,224)
                volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
                # strde=12
                stride = volume_batch_r.shape[0] // 2
                # (96,4,224,224)
                preds = torch.zeros([stride * T, self.num_classes, w, h]).cuda()
                for i in range(T // 2):
                    ema_inputs = volume_batch_r + \
                                 torch.clamp(torch.randn_like(
                                     volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[2 * stride * i:2 * stride *
                                             (i + 1)] = self.ema_model(ema_inputs)
                # (8,12,4,224,224)
                preds_copy = preds
                preds = F.softmax(preds, dim=1)

                preds = preds.reshape(T, stride, self.num_classes, w, h)
                # (12,4,224,224)
                preds = torch.mean(preds, dim=0)
                # (12,1,224,224)
                uncertainty = -1.0 * \
                              torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)

                # loss_ce = ce_loss(outputs[:args.labeled_bs],
                #                   label_batch[:args.labeled_bs][:].long())
                # loss_dice = dice_loss(
                #     outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
                # 监督损失
                # supervised_loss = 0.5 * (loss_dice + loss_ce)

                consistency_dist = softmax_mse_loss(
                    outputs[4:], ema_output)  # (batch, 2, 112,112,80)
                # 阈值
                # threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num,
                #                                                 max_iterations)) * np.log(2)
                threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num,
                                                                self.local_epoch)) * np.log(2)
                mask = (uncertainty < threshold).float()
                consistency_loss = torch.sum(
                    mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
                consistency_weight = get_current_consistency_weight(iter_num // 150)

                # # (24,4,224,224)
                # with torch.no_grad():
                #     # 无监督学习中的输出
                #     ema_output = self.ema_model(ema_inputs)
                # T = 8
                # # (12,3,224,224)
                # _, _, w, h = unlabeled_volume_batch.shape
                # # (24,3,224,224)
                # volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
                # # strde=12
                # stride = volume_batch_r.shape[0] // 2
                # # (56,2,224,224)
                # preds = torch.zeros([stride * T, self.num_classes, w, h]).cuda()
                # for i in range(T // 2):
                #     ema_inputs = volume_batch_r + \
                #                  torch.clamp(torch.randn_like(
                #                      volume_batch_r) * 0.1, -0.2, 0.2)
                #     with torch.no_grad():
                #         preds[2 * stride * i:2 * stride *
                #                              (i + 1)] = self.ema_model(ema_inputs)
                # # (56,2,224,224)
                # preds = F.softmax(preds, dim=1)
                # #(8,7,2,224,224)
                # preds = preds.reshape(T, stride, self.num_classes, w, h)
                # # (7,2,224,224)
                # #求T次平均
                # preds = torch.mean(preds, dim=0)
                # # (12,1,224,224)
                # #求不确定性
                # uncertainty = -1.0 * \
                #               torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
                #
                # consistency_weight = get_current_consistency_weight(iter_num // 5)
                # #一致性损失
                # consistency_dist = softmax_mse_loss(
                #     outputs[1:], ema_output)  # (batch, 2, 112,112,80)
                # # 阈值
                # threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num,
                #                                                 self.local_epoch)) * np.log(2)
                # mask = (uncertainty < threshold).float()
                # consistency_loss = torch.sum(
                #     mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

                #loss = consistency_weight * consistency_loss
                loss = consistency_loss * consistency_weight * 1000
                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                update_ema_variables(self.model, self.ema_model, self.ema_decay, iter_num)
                batch_loss.append(loss.item())
                iter_num = iter_num + 1
                if self.device == "cuda": torch.cuda.empty_cache()
            iter_loss.append(sum(batch_loss) / len(batch_loss))
        self.model.to("cpu")
        loss = sum(iter_loss) / len(iter_loss)
        return loss

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, dice_pred = 0, 0
        dice_predmask = 0
        with torch.no_grad():
            for dataset, labelName in self.valdataloader:
                # data, labels = data.float().to(self.device), labels.long().to(self.device)
                data = dataset['image'].float().to(self.device)
                labels = dataset['label'].float().to(self.device)
                outputs = self.model(data)
                #print(outputs)
                outputssoft = F.softmax(outputs, dim=1)
                img_path = './un_img/'
                save_on_batch(data, labels, outputs, labelName, img_path)
                test_loss += eval(self.criterion)(2)(outputssoft, labels.unsqueeze(1)).item()
              #  test_loss += eval(self.criterion)()(outputs, labels.unsqueeze(1)).item()

                # predicted = outputs.argmax(dim=1, keepdim=True)
                # correct += predicted.eq(labels.view_as(predicted)).sum().item()
                #dice_pred_t = torch.tensor(dice(outputs, labels)).cuda()
                # dice_pred_t = dice(outputs, labels)
                # dice_pred += dice_pred_t
                #dice_predmask =dice_pred_tmask+dice_predmask
                # iou_pred += iou_pred_t
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.valdataloader)
        test_dice = 1 - test_loss
        #dice_pred = dice_predmask / len(self.valdataloader)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> test_dice: {100. * test_dice:.2f}%\n"
        print(message, flush=True);
        logging.info(message)
        del message;
        gc.collect()

        return test_loss, test_dice
