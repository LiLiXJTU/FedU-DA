import copy
import gc
import logging
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


class Local(object):
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

    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={},
                 optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        self.model = eval(model_config["name"])(**model_config)

        self.seed = global_config["seed"]
        self.device = global_config["device"]
        #self.mp_flag = global_config["is_mp"]
        self.mp_flag = False

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]

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

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

        # split local dataset for each client
        # 原来的
        # local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid)

        UESTC, CTseg, MosMed, medseg, test_dataset, val_UESTC, val_CTseg, val_MosMed, val_medseg = LoadDatasets(
            self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid)
        # local_datasets, test_dataset = datasets(self.data_path, self.dataset_name, self.num_clients,self.num_shards, self.iid)
        # assign dataset to each client
        # self.clients = self.create_clients(local_datasets) #--------------------------------------------------------------------
        self.clients = self.create_clients(UESTC, CTseg, MosMed, medseg, val_UESTC, val_CTseg, val_MosMed, val_medseg)
        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # configure detailed settings for client upate and
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config
        )


    # def create_clients(self, local_datasets):
    def create_clients(self, UESTC, CTseg, MosMed, medseg, val_UESTC, val_CTseg, val_MosMed, val_medseg):
        """Initialize each Client instance."""
        clients = []
        # 创建客户端，传数据进去，可以考虑分别传数据进去
        client0 = Client(client_id=0, local_data=UESTC, local_valdata=val_UESTC, device=self.device)
        clients.append(client0)
        # client1 = UnsupervisedClient(client_id=1, local_data=CTseg, local_valdata=val_CTseg, device=self.device)
        # clients.append(client1)
        # client2 = UnsupervisedClient(client_id=2, local_data=MosMed, local_valdata=val_MosMed, device=self.device)
        # clients.append(client2)
        # client3 = UnsupervisedClient(client_id=3, local_data=medseg, local_valdata=val_medseg, device=self.device)
        # clients.append(client3)
        # for k, dataset in tqdm(enumerate(local_datasets), leave=False):
        #     client = Client(client_id=k, local_data=dataset, device=self.device)
        #     clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()


    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()
        # 随机选择客户端数量，max[c*num,1],我们只有一个
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        # 转化成list
        sampled_client_indices = sorted(
            np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices

    def update_selected_clients(self, sampled_client_indices):
        print('------update_selected_clients')
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].model = copy.deepcopy(self.model)
            train_loss = self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})! train_loss:{str(train_loss)}"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

        return selected_total_size


    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()


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
            print(message);
            logging.info(message)
            del message;
            gc.collect()

            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        else:
            self.evaluate_selected_models(sampled_client_indices)

    def fit(self):
        """Execute the whole process of the federated learning."""
        # self.results = {"loss": [], "dice": []}
        self.train_federated_model()
        # for r in range(self.num_rounds):
        #     self._round = r + 1
        #
        #     self.train_federated_model()
        #     test_loss, test_dice = self.evaluate_global_model()
        #
        #     self.results['loss'].append(test_loss)
        #     self.results['dice'].append(test_dice)
        #
        #     self.writer.add_scalars(
        #         'Loss',
        #         {
        #             f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
        #         self._round
        #     )
        #     self.writer.add_scalars(
        #         'Dice',
        #         {
        #             f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_dice},
        #         self._round
        #     )
        #
        #     message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
        #         \n\t[Server] ...finished evaluation!\
        #         \n\t=> Loss: {test_loss:.4f}\
        #         \n\t=> Dice: {100. * test_dice:.2f}%\n"
        #     print(message);
        #     logging.info(message)
        #     del message;
        #     gc.collect()
