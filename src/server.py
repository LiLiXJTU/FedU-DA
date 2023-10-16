import copy
import gc
import logging
from PIL import Image
from local.fusiontrans import UFusionNet
from src.loss import DiceLoss, MultiClassDiceLoss
import numpy as np
import torch
import torch.nn as nn
import random
from torch.backends import cudnn
from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from .LoadData import LoadDatasets
from .models import *
from .un_client import UnsupervisedClient
from .utils import *
from .client import Client

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        self.model = eval(model_config["name"])(**model_config)
        # self.ema_model = eval(model_config["name"])(**model_config)
        #self.model = UFusionNet(**model_config)
        
        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]
        self.save_img_path = data_config["save_img_path"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]

        self.criterion = fed_config["criterion"]
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optim_config["lr"],
        #                              betas=(0.9, 0.999), weight_decay=5e-4)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=optim_config["lr"],momentum=optim_config["momentum"])

        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config
        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        deterministic = True
        if not deterministic:
            cudnn.benchmark = True
            cudnn.deterministic = False
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        init_net(self.model, **self.init_config)
        # init_net(self.ema_model, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        #原来的
        #local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid)

        A_DATA, B_DATA, C_DATA, test_dataset, val_A,val_B, val_C = LoadDatasets()
        #local_datasets, test_dataset = datasets(self.data_path, self.dataset_name, self.num_clients,self.num_shards, self.iid)
        # assign dataset to each client
        #self.clients = self.create_clients(local_datasets) #--------------------------------------------------------------------
        self.clients = self.create_clients(A_DATA, B_DATA, C_DATA, val_A,val_B, val_C)
        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.max_dice=[0,0,0,0]
        self.val_dice=[0,0,0,0]
        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config
            )
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    # def create_clients(self, local_datasets):
    def create_clients(self, A_DATA, B_DATA, C_DATA, val_A,val_B, val_C):
        """Initialize each Client instance."""
        clients = []
        #创建客户端，传数据进去，可以考虑分别传数据进去
        client0 = Client(client_id=0,local_data = A_DATA, local_valdata = val_A, device=self.device,save_img_path = self.save_img_path)
        clients.append(client0)
        client1 = Client(client_id=1,local_data = B_DATA, local_valdata = val_B, device=self.device,save_img_path  = self.save_img_path)
        clients.append(client1)
        client2 = Client(client_id=2,local_data = C_DATA, local_valdata = val_C, device=self.device,save_img_path = self.save_img_path)
        clients.append(client2)
        # for k, dataset in tqdm(enumerate(local_datasets), leave=False):
        #     client = Client(client_id=k, local_data=dataset, device=self.device)
        #     clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)
                #client.ema_model = self.ema_model

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
                #self.clients[idx].ema_model = self.ema_model
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        #随机选择客户端数量，max[c*num,1],我们只有一个
        num_sampled_clients = max(int(self.fraction * self.num_clients), 3)
        #num_sampled_clients = random.randint(1, 3)
        #转化成list
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        print('------update_selected_clients')
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            train_loss = self.clients[idx].client_update(idx)
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})! train_loss:{str(train_loss)}"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    def mp_update_selected_clients(self, selected_index):
        print('mp_update_selected_clients')
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        train_loss,optimizer = self.clients[selected_index].client_update(selected_index)
        print(selected_index,'selected_index')
        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})! train_loss:{str(train_loss)}"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        #原始的
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        #除了客户端0的outc不传
        # averaged_weights = OrderedDict()
        # for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
        #     local_weights = self.clients[idx].model.state_dict()
        #     for key,name in enumerate(self.model.state_dict()):
        #         if 'outc' not in name:
        #             if it == 0:
        #                 averaged_weights[name] = coefficients[it] * local_weights[name]
        #             else:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #         else:
        #             if it == 0:
        #                 averaged_weights[name] = 0 * local_weights[name]
        #             else:
        #                 averaged_weights[name] += 0.5 * local_weights[name]
        # self.model.load_state_dict(averaged_weights,strict=False)

        #只传可训练的参数
        # for param in self.model.parameters():
        #     param.data.zero_()
        # for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
        #     for (server_paramname,server_param), (client_paramname,client_param) in zip(self.model.named_parameters(), self.clients[idx].model.named_parameters()):
        #         if server_param.is_cuda is True:
        #             server_param = server_param.cpu()
        #         if 'norm' not in client_paramname:
        #             # print('server:',server_param.is_cuda)
        #             # print(client_param.is_cuda)
        #             server_param.data += client_param.data.clone() * coefficients[it]
        #             #print(server_param)
        #         if 'norm' in client_paramname:
        #             print('not submit bn!')

        #除了bn层的方差和标准差，其他都传
        # averaged_weights = OrderedDict()
        # for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
        #     local_weights = self.clients[idx].model.state_dict()
        #     for key,name in enumerate(self.model.state_dict()):
        #         if 'running_mean' not in name:
        #             if 'running_var' not in name:
        #                 if it == 0:
        #                     averaged_weights[name] = coefficients[it] * local_weights[name]
        #                 else:
        #                     averaged_weights[name] += coefficients[it] * local_weights[name]
        # self.model.load_state_dict(averaged_weights,strict=False)

        #除了bn层，其他都传
        # averaged_weights = OrderedDict()
        # for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
        #     local_weights = self.clients[idx].model.state_dict()
        #     if it == 0:
        #         for key,name in enumerate(self.model.state_dict()):
        #             if 'inc' in name:
        #                 averaged_weights[name] = coefficients[it] * local_weights[name]
        #             elif 'down1' in name:
        #                 averaged_weights[name] = coefficients[it] * local_weights[name]
        #             elif 'down2' in name:
        #                 averaged_weights[name] = coefficients[it] * local_weights[name]
        #             elif 'down3' in name:
        #                 averaged_weights[name] = coefficients[it] * local_weights[name]
        #             elif 'down4' in name:
        #                 averaged_weights[name] = coefficients[it] * local_weights[name]
        #             else:
        #                 averaged_weights[name] = 0.5 * local_weights[name]
        #     if it == 1:
        #         for key, name in enumerate(self.model.state_dict()):
        #             if 'inc' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #             elif 'down1' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #             elif 'down2' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #             elif 'down3' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #             elif 'down4' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #             else:
        #                 averaged_weights[name] += 0.5 * local_weights[name]
        #     if it == 2:
        #         for key, name in enumerate(self.model.state_dict()):
        #             if 'inc' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #             elif 'down1' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #             elif 'down2' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #             elif 'down3' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #             elif 'down4' in name:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        #
        #
        #         if 'norm' not in name:
        #             if it == 0:
        #                 averaged_weights[name] = coefficients[it] * local_weights[name]
        #             else:
        #                 averaged_weights[name] += coefficients[it] * local_weights[name]
        # self.model.load_state_dict(averaged_weights,strict=False)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print('1111')
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        print('2222')
        print(self.max_dice[selected_index])
        val_loss,self.val_dice[selected_index]=self.clients[selected_index].client_evaluate()
        print(self.val_dice[selected_index])
        if self.val_dice[selected_index] > self.max_dice[selected_index]:
            print('333')
            logger.info(
                '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(self.max_dice[selected_index], self.val_dice[selected_index]))
            self.max_dice[selected_index] = self.val_dice[selected_index]
            save_path='./fedbestmodels/' + str(selected_index) + '/'
            save_checkpoint({
                            'best_model': True,
                            'model': selected_index,
                            'state_dict': self.clients[selected_index].model.state_dict(),
                            'val_loss': val_loss}, save_path)
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        if self.mp_flag:
            message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
            print(message); logging.info(message)
            del message; gc.collect()

            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        else:
            self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)
        test_loss, dice_pred = 0, 0
        test_dice = 0
        with torch.no_grad():
            for dataset, labelName in self.dataloader:
                # data, labels = data.float().to(self.device), labels.long().to(self.device)
                data = dataset['image'].float().to(self.device)
                labels = dataset['label'].float().to(self.device)
                outputs = self.model(data)
                mask = torch.argmax(outputs, dim=1)
                outputssoft = torch.softmax(outputs, dim=1)
                img_path = self.save_img_path + 'global_img/'
                for i in range(mask.shape[0]):
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

                    # mask_arr = sItk.GetImageFromArray(mask_arr)
                    # sItk.WriteImage(mask_arr, img_path + labelName[i][:-4] + ".nii")
                #outputs = outputs.data.max(1)[1]
                #outputs = torch.max(outputs, dim=1).values.unsqueeze(1)
                #outputs = torch.argmax(outputs, dim=1)
                idx = 999
                weights = [1,1,1,1,1]
                dice_loss, dice_score= MultiClassDiceLoss()(outputssoft, labels,idx,weights)
                dice_loss = dice_loss.item()
                dice_score = dice_score.item()

                test_loss = test_loss + dice_loss
                test_dice = test_dice + dice_score


                # predicted = outputs.argmax(dim=1, keepdim=True)
                # correct += predicted.eq(labels.view_as(predicted)).sum().item()

                # dice_pred_t = dice_show(outputs, labels)
                # dice_pred += dice_pred_t

                # iou_pred += iou_pred_t
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        #test_dice = dice_pred / len(self.dataloader)
        test_dice = dice_score / len(self.dataloader)
        return test_loss, test_dice

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "dice": []}
        model_path = './global_model'
        max_dice = 0
        min_loss = 999
        for r in range(self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()
            test_loss, test_dice = self.evaluate_global_model()
            
            self.results['loss'].append(test_loss)
            self.results['dice'].append(test_dice)
            model_r_path = model_path+'/'+str(r)
            if r + 1 > 0:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, test_dice))
                min_loss = test_loss
                save_checkpoint({'epoch': r,
                                 'best_model': True,
                                 'model': 'UNet',
                                 'state_dict': self.model.state_dict(),
                                 'test_loss': test_loss}, model_r_path)
            # if test_loss < min_loss:
                # if r + 1 > 0:
                #     logger.info(
                #         '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, test_dice))
                #     min_loss = test_loss
                #     save_checkpoint({'epoch': r,
                #                      'best_model': True,
                #                      'model':'UNet',
                #                      'state_dict': self.model.state_dict(),
                #                      'test_loss': test_loss}, model_path)
            self.writer.add_scalars(
                'Loss',
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
                self._round
                )
            self.writer.add_scalars(
                'Dice',
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_dice},
                self._round
                )

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Dice: {100. * test_dice:.2f}%\n"
            print(message); logging.info(message)
            del message; gc.collect()
        self.transmit_model()
def dice_show(preds, masks):
    # preds[preds >= 0.5] = 1
    # preds[preds < 0.5] = 0
    # # print("2",np.sum(tmp))
    # targets = masks.float()
    # targets[targets > 0] = 1
    # targets[targets <= 0] = 0
    # dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
    # train_dice = 1.0 - dice_loss(preds, targets)
    # return train_dice
    smooth = 1e-5
    intersection = (preds * masks).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + masks.sum() + smooth)
    return dice

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model)
    torch.save(state, filename)

    #torch.save({'model': lily.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i, 'loss_array': loss_list, 'acc_array': acc_list}, '../model_dict/resnet50model_with_pretrain.dict')
