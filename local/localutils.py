import numpy as np
import torch
from sklearn.metrics import roc_auc_score,jaccard_score
import cv2
from torch import nn
import torch.nn.functional as F
import math
from functools import wraps
import warnings
import weakref
from torch.optim.optimizer import Optimizer
from math import exp

class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()

        return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
        # t = w*(t*2-1)
        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)
        # print "------",dice.data

        loss = dice.mean()
        return loss

class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        DICE_LOSS = self.dice_weight * dice
        BCE_LOSS = self.BCE_weight * BCE
        dice_BCE_loss = DICE_LOSS+BCE_LOSS

        return dice_BCE_loss,DICE_LOSS,BCE_LOSS

def auc_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    aucs = []
    for i in range(pred.shape[1]):
        prediction = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask = masks[i].cpu().detach().numpy()
        # print("rrr",np.max(mask), np.min(mask))
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)

def iou_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    ious = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp>=0.5] = 1
        pred_tmp[pred_tmp<0.5] = 0
        # print("2",np.sum(tmp))
        mask_tmp[mask_tmp>0] = 1
        mask_tmp[mask_tmp<=0] = 0
        # print("rrr",np.max(mask), np.min(mask))
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    return np.mean(ious)

def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    dices = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp>=0.5] = 1
        pred_tmp[pred_tmp<0.5] = 0
        # print("2",np.sum(tmp))
        mask_tmp[mask_tmp>0] = 1
        mask_tmp[mask_tmp<=0] = 0
        # print("rrr",np.max(mask), np.min(mask))
        dices.append(dice_coef(mask_tmp, pred_tmp))
    return np.mean(dices)

def save_on_batch(images1, masks, pred, names, vis_path):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp>=0.5] = 255
        pred_tmp[pred_tmp<0.5] = 0
        mask_tmp[mask_tmp>0] = 255
        mask_tmp[mask_tmp<=0] = 0

        cv2.imwrite(vis_path+ names[i][:-4]+"_pred.jpg", pred_tmp)
        cv2.imwrite(vis_path+names[i][:-4]+"_gt.jpg", mask_tmp)



class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
        #高斯滤波求均值
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    img2 = img2.unsqueeze(1)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    #平方
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    #在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
    #方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma1 = torch.sqrt(sigma1_sq)
    sigma2 = torch.sqrt(sigma2_sq)
    #协方差
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    #对比度
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    s1 = sigma12 + C2 / 2
    s2 = sigma1*sigma2 + C2 / 2
    ssim_map = s1 / s2
    #ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        #print("sim",sim)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)

class DiceMSSSIM(nn.Module):
    def __init__(self,dice_weight=1,MSSSIM_weight=1):
        super(DiceMSSSIM, self).__init__()
        self.MSSSIM_loss = MSSSIM(window_size=11, size_average=True, channel=3)
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.MSSSIM_weight = MSSSIM_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        MSSSIM = self.MSSSIM_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        dice_MSSSIM_loss = self.dice_weight * dice + self.MSSSIM_weight * MSSSIM

        return dice_MSSSIM_loss

class WeightedDiceBCEMSSSIM(nn.Module):
        def __init__(self, dice_weight=0.3, MSSSIM_weight=0.4,BCE_weight=0.3):
            super(WeightedDiceBCEMSSSIM, self).__init__()
            self.MSSSIM_loss = MSSSIM(window_size=11, size_average=True, channel=3)
            self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
            self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
            self.BCE_weight = BCE_weight
            self.MSSSIM_weight = MSSSIM_weight
            self.dice_weight = dice_weight

        def _show_dice(self, inputs, targets):
            inputs[inputs >= 0.5] = 1
            inputs[inputs < 0.5] = 0
            # print("2",np.sum(tmp))
            targets[targets > 0] = 1
            targets[targets <= 0] = 0
            hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
            return hard_dice_coeff

        def forward(self, inputs, targets):
            # inputs = inputs.contiguous().view(-1)
            # targets = targets.contiguous().view(-1)
            # print ("dice_loss", self.dice_loss(inputs, targets))
            # print ("focal_loss", self.focal_loss(inputs, targets))
            BCE = self.BCE_loss(inputs, targets)
            dice = self.dice_loss(inputs, targets)
            MSSSIM = self.MSSSIM_loss(inputs, targets)
            # print "dice",dice
            # print "focal",focal
            dice_MSSSIM_loss = self.dice_weight * dice + self.MSSSIM_weight * MSSSIM+self.BCE_weight * BCE

            return dice_MSSSIM_loss



def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

# def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
#     """
#     copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
#     net_output must be (b, c, x, y(, z)))
#     gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
#     if mask is provided it must have shape (b, 1, x, y(, z)))
#     :param net_output:
#     :param gt:
#     :param axes:
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(net_output.size())))
#
#     shp_x = net_output.shape
#     shp_y = gt.shape
#
#     with torch.no_grad():
#         if len(shp_x) != len(shp_y):
#             gt = gt.view((shp_y[0], 1, *shp_y[1:]))
#
#         if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
#             # if this is the case then gt is probably already a one hot encoding
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(shp_x)
#             if net_output.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(net_output.device.index)
#             y_onehot.scatter_(1, gt, 1)
#
#     tp = net_output * y_onehot
#     fp = net_output * (1 - y_onehot)
#     fn = (1 - net_output) * y_onehot
#
#     if mask is not None:
#         tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
#         fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
#         fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
#
#     if square:
#         tp = tp ** 2
#         fp = fp ** 2
#         fn = fn ** 2
#
#     tp = sum_tensor(tp, axes, keepdim=False)
#     fp = sum_tensor(fp, axes, keepdim=False)
#     fn = sum_tensor(fn, axes, keepdim=False)
#
#     return tp, fp, fn

class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, preds, target):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        #bound相当于就是preds
        net_output = softmax_helper(preds)
        # print('net_output shape: ', net_output.shape)
        pc = net_output[:,:, ...].type(torch.float32)
        #bound
        target = torch.unsqueeze(target, 1)
        dc = target[:,:, ...].type(torch.float32)

        multipled = torch.einsum("bcxy,bcxy->bcxy", pc, dc)
        bd_loss = multipled.mean()

        return bd_loss

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        #one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        #target = torch.Tensor(target).long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,num_classes)

        return super(CrossentropyND, self).forward(inp, target)

class TopKLoss(CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        #target = target[:, 0].long()
        #target = torch.Tensor(target).long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

class Dice_TopK_BD(nn.Module):
    def __init__(self,dice_weight=0.02,TopK_weight=0.4,BD_weight=0.58):
        super(Dice_TopK_BD, self).__init__()
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.TopK_loss = TopKLoss(weight=None, ignore_index=-100, k=10)
        self.BD_loss = BDLoss()
        self.dice_weight = dice_weight
        self.TopK_weight = TopK_weight
        self.BD_weight = BD_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BD = self.BD_loss(inputs, targets)
        TopK = self.TopK_loss(inputs, targets)

        print("dice",dice)
        print("TopK",TopK)
        print("BD", BD)
        Dice_TopK_BD_loss = self.dice_weight * dice + self.TopK_weight * TopK + self.BD_weight * BD

        return Dice_TopK_BD_loss

class Dice_BD(nn.Module):
    def __init__(self,dice_weight=0.02,BD_weight=0.98):
        super(Dice_BD, self).__init__()
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BD_loss = BDLoss()
        self.dice_weight = dice_weight
        self.BD_weight = BD_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BD = self.BD_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        Dice_BD_loss = (self.dice_weight * dice + self.BD_weight * BD) *30

        return Dice_BD_loss
