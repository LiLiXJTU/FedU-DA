from random import random
import torch
import torch.nn.functional as F
import numpy as np
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
def client_update(self):
    """Update local model using local dataset."""
    self.batch_size = 16
    self.num_classes = 2
    self.ema_decay = 0.99
    self.seed = 500
    torch.manual_seed(self.seed)
    random.seed(self.seed)
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.cuda.manual_seed(self.seed)
    torch.cuda.manual_seed_all(self.seed)
    self.model.train()
    self.model.to(self.device)
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
    iter_loss = []
    iter_num = 0
    for iter in range(self.local_epoch):
        for dataset, labelName in self.dataloader:
            # data, labels = data.float().to(self.device), labels.long().to(self.device)
            data = dataset['image'].float().to(self.device)
            # labels = dataset['label'].float().to(self.device)
            # 选取最小的4个
            unlabeled_volume_batch = data[4:]
            inputs = data
            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            outputs = self.model(inputs)
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
            preds = F.softmax(preds, dim=1)

            preds = preds.reshape(T, stride, self.num_classes, w, h)
            # (12,4,224,224)
            preds = torch.mean(preds, dim=0)
            # (12,1,224,224)
            uncertainty = -1.0 * \
                          torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            consistency_dist = softmax_mse_loss(
                outputs[4:], ema_output)  # (batch, 2, 112,112,80)
            # 阈值
            threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num,
                                                      self.local_epoch)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(
                mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            loss = consistency_loss
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

def train(self, args, net_w, op_dict, epoch, unlabeled_idx, train_dl_local, n_classes):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.train()
        self.ema_model.eval()

        self.model.cuda()
        self.ema_model.cuda()

        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.unsup_lr

        self.epoch = epoch
        if self.flag:
            self.ema_model.load_state_dict(copy.deepcopy(net_w))
            self.flag = False
            logging.info('EMA model initialized')

        epoch_loss = []
        logging.info('Unlabeled client %d begin unsupervised training' % unlabeled_idx)
        correct_pseu = 0
        all_pseu = 0
        test_right = 0
        test_right_ema = 0
        train_right = 0
        same_total = 0
        for epoch in range(args.local_ep):
            batch_loss = []

            for i, (_, weak_aug_batch, label_batch) in enumerate(train_dl_local):
                weak_aug_batch = [weak_aug_batch[version].cuda() for version in range(len(weak_aug_batch))]
                with torch.no_grad():
                    guessed = label_guessing(self.ema_model, [weak_aug_batch[0]], args.model)
                    sharpened = sharpen(guessed)

                pseu = torch.argmax(sharpened, dim=1)
                label = label_batch.squeeze()
                if len(label.shape) == 0:
                    label = label.unsqueeze(dim=0)

                correct_pseu += torch.sum(label[torch.max(sharpened, dim=1)[0] > args.confidence_threshold] == pseu[
                    torch.max(sharpened, dim=1)[0] > args.confidence_threshold].cpu()).item()
                all_pseu += len(pseu[torch.max(sharpened, dim=1)[0] > args.confidence_threshold])
                train_right += sum([pseu[i].cpu() == label_batch[i].int() for i in range(label_batch.shape[0])])

                logits_str = self.model(weak_aug_batch[1], model=args.model)[2]
                probs_str = F.softmax(logits_str, dim=1)
                pred_label = torch.argmax(logits_str, dim=1)

                same_total += sum([pred_label[sam] == pseu[sam] for sam in range(logits_str.shape[0])])

                loss_u = torch.sum(losses.softmax_mse_loss(probs_str, sharpened)) / args.batch_size

                ramp_up_value = self.ramp_up(current=self.epoch)

                loss = ramp_up_value * args.lambda_u * loss_u
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_grad_norm)
                self.optimizer.step()

                update_ema_variables(self.model, self.ema_model, args.ema_decay, self.iter_num)

                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.epoch = self.epoch + 1
        self.model.cpu()
        self.ema_model.cpu()
        return self.model.state_dict(), self.ema_model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(
            self.optimizer.state_dict()), ramp_up_value, correct_pseu, all_pseu, test_right, train_right.cpu().item(), test_right_ema, same_total.cpu().item()


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.batch_size, shuffle=True)
        net = DenseNet121(out_size=5, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[0, 1])
        self.ema_model = net.cuda()
        for param in self.ema_model.parameters():
            param.detach_()
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.base_lr = 2e-4

    def train(self, args, net, op_dict, epoch, target_matrix):
        net.train()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        self.epoch = epoch
        if self.flag:
            self.ema_model.load_state_dict(net.state_dict())
            self.flag = False
            print('done')

        epoch_loss = []
        print('begin training')
        for epoch in range(args.local_ep):

            batch_loss = []
            iter_max = len(self.ldr_train)

            for i, (_, _, (image_batch, ema_image_batch), label_batch) in enumerate(self.ldr_train):

                image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()

                ema_inputs = ema_image_batch
                inputs = image_batch

                _, outputs = net(inputs)

                with torch.no_grad():
                    ema_activations, ema_output = self.ema_model(ema_inputs)
                T = 10

                with torch.no_grad():
                    _, logits_sum = net(inputs)
                    for i in range(T):
                        _, logits = net(inputs)
                        logits_sum = logits_sum + logits
                    logits = logits_sum / (T + 1)
                    preds = F.softmax(logits, dim=1)
                    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1)
                    uncertainty_mask = (uncertainty < 2.0)

                with torch.no_grad():
                    activations = F.softmax(outputs, dim=1)
                    confidence, _ = torch.max(activations, dim=1)
                    confidence_mask = (confidence >= 0.3)
                mask = confidence_mask * uncertainty_mask

                pseudo_labels = torch.argmax(activations[mask], dim=1)
                pseudo_labels = F.one_hot(pseudo_labels, num_classes=5)
                source_matrix = get_confuse_matrix(outputs[mask], pseudo_labels)

                consistency_weight = get_current_consistency_weight(self.epoch)
                consistency_dist = torch.sum(losses.softmax_mse_loss(outputs, ema_output)) / args.batch_size
                consistency_loss = consistency_dist

                loss = 15 * consistency_weight * consistency_loss + 15 * consistency_weight * torch.sum(
                    kd_loss(source_matrix, target_matrix))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                update_ema_variables(net, self.ema_model, args.ema_decay, self.iter_num)
                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            timestamp = get_timestamp()

            self.epoch = self.epoch + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(epoch_loss)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())