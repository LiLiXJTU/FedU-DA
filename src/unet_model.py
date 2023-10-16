import torch.nn as nn
import torch
import numpy as np
import torch
from torch.distributions.uniform import Uniform
#################################
# Models for federated learning #
#################################
# McMahan et al., 2016; 199,210 parameters
class TwoNN(nn.Module):
    def __init__(self, name, in_features, num_hiddens, num_classes):
        super(TwoNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.fc1 = nn.Linear(in_features=in_features, out_features=num_hiddens, bias=True)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_hiddens, bias=True)
        self.fc3 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# McMahan et al., 2016; 1,663,370 parameters
class CNN(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (7 * 7), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# for CIFAR10
class CNN2(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN2, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (8 * 8), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out
class ConvB(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)
        return out

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        #..................................................................................
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, in_channels//2, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DynamicConv2d, self).__init__()
        self.num_branches = 5
        self.branches = nn.ModuleList()
        for i in range(self.num_branches):
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels//5, out_channels//5, kernel_size=1),
                nn.BatchNorm2d(out_channels//5),
                nn.ReLU(),
                nn.Conv2d(out_channels//5, out_channels//5, kernel_size=kernel_size,padding=1),
                nn.BatchNorm2d(out_channels//5),
                nn.ReLU()
            ))

    def forward(self, x):
        # split the input tensor into 5 parts along the channel dimension
        x_split = torch.chunk(x, self.num_branches, dim=1)
        # apply different convolutions to each part
        out = []
        for i in range(self.num_branches):
            out.append(self.branches[i](x_split[i]))
        # concatenate the outputs from the different branches along the channel dimension
        out = torch.cat(out, dim=1)
        return out

class UNet(nn.Module):
    def __init__(self,name, in_channels, num_classes):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.name = name
        # Question here
        n_channels = 32
        self.inc = ConvB(in_channels, n_channels)
        self.down1 = DownBlock(n_channels, n_channels*2, nb_Conv=2)
        self.down2 = DownBlock(n_channels*2, n_channels*4, nb_Conv=2)
        self.down3 = DownBlock(n_channels*4, n_channels*8, nb_Conv=2)
        self.down4 = DownBlock(n_channels*8, n_channels*16, nb_Conv=2)
        self.up4 = UpBlock(n_channels*16, n_channels*4, nb_Conv=2)
        self.up3 = UpBlock(n_channels*8, n_channels*2, nb_Conv=2)
        self.up2 = UpBlock(n_channels*4, n_channels, nb_Conv=2)
        self.up1 = UpBlock(n_channels*2, n_channels, nb_Conv=2)
        self.outc = nn.Conv2d(n_channels, num_classes, kernel_size=(1,1))
        #self.DynamicConv2d = DynamicConv2d(num_classes,num_classes)
        if num_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    #     self.cls = nn.Sequential(
    #         nn.Dropout(p=0.5),  # p为不保留节点数的比例。
    #         nn.Conv2d(n_channels*16, 5, 1),  # Conv2d的参数：输入的特征通道，卷积核尺寸，步长
    #         nn.AdaptiveMaxPool2d(1),  # 自适应最大池化，参数为（H,W）或只有一个H，表示输出信号的尺寸。输出的尺寸不变，后两个维度变为参数大小。
    #         nn.Softmax())
    #
    # def dotProduct(self, seg, cls):
    #     B, N, H, W = seg.size()  # seg是传入的深度卷积结果，是矩阵。
    #     seg = seg.view(B, N, H * W)  # view和reshape作用一样，重新定义矩阵的性状。
    #     final = torch.einsum("ijk,ij->ijk", [seg, cls])  # 利用爱因斯坦求和约定方法求乘积的和。
    #     final = final.view(B, N, H, W)
    #     return final



    def forward(self, x):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #cls_branch = self.cls(x5).squeeze(3).squeeze(2)
        # (B,N,1,1)->(B,N)  #操作(dropout, conv1*1, adaptiveMaxPool, Sigmoid)后，产生一个二维张量  squeeze(x)只有当维度x的值为1时，才能去掉该维度。
        #cls_branch_max = cls_branch.argmax(dim=1)
        # dim=1将1维去掉，返回最大值对应的索引。  通过argmax，分类结果被转为一个单一数字输出。  #argmax(a, axis=None, out=Nont):a为输入的数组；axis=0按列寻找，axis=1按行寻找最大值对应的索引；out结果将被插入到a中。
        #cls_branch_max = cls_branch_max[:, np.newaxis].float()  # 在np.newaxis的位置增加一个维度，故此时是增加一个列维度。

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print("111")
        else:
            logits = self.outc(x)
        #logits = self.dotProduct(logits, cls_branch)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        #logits = self.DynamicConv2d(logits)
        return logits


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2,1,224,224)).cuda()
    model = UNet('unet',1,5).cuda()
    #param = count_param(model)
    y = model(x)
    print('Output shape:',y.shape)
    #print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
    param1 = sum([param.nelement() for param in model.parameters()])
    # param = count_param(model)
    y = model(x)
    # print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
    print(param1)

# def kaiming_normal_init_weight(model):
#     for m in model.modules():
#         if isinstance(m, nn.Conv3d):
#             torch.nn.init.kaiming_normal_(m.weight)
#         elif isinstance(m, nn.BatchNorm3d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#     return model
#
# def sparse_init_weight(model):
#     for m in model.modules():
#         if isinstance(m, nn.Conv3d):
#             torch.nn.init.sparse_(m.weight, sparsity=0.1)
#         elif isinstance(m, nn.BatchNorm3d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#     return model
#
#
# class ConvBlock(nn.Module):
#     """
# two
# convolution
# layers
# with batch norm and leaky relu"""
#
#     def __init__(self, in_channels, out_channels, dropout_p):
#         super(ConvBlock, self).__init__()
#         # self.conv_conv = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(out_channels),
#         #     nn.LeakyReLU(),
#         #     nn.Dropout(dropout_p),
#         #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(out_channels),
#         #     nn.LeakyReLU()
#         # )
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.norm = nn.BatchNorm2d(out_channels)
#         self.conv_conv = nn.Sequential(
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_p),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         )
#         self.ac = nn.LeakyReLU()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm(x)
#         x = self.conv_conv(x)
#         x = self.norm(x)
#         x = self.ac(x)
#         return x
#
#
# class DownBlock(nn.Module):
#     """Downsampling followed by ConvBlock"""
#
#     def __init__(self, in_channels, out_channels, dropout_p):
#         super(DownBlock, self).__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             ConvBlock(in_channels, out_channels, dropout_p)
#
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
#
# class UpBlock(nn.Module):
#     """Upssampling followed by ConvBlock"""
#
#     def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
#                  bilinear=True):
#         super(UpBlock, self).__init__()
#         self.bilinear = bilinear
#         if bilinear:
#             self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
#             self.up = nn.Upsample(
#                 scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(
#                 in_channels1, in_channels2, kernel_size=2, stride=2)
#         self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)
#
#     def forward(self, x1, x2):
#         if self.bilinear:
#             x1 = self.conv1x1(x1)
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
#
#
# class Encoder(nn.Module):
#     def __init__(self, params):
#         super(Encoder, self).__init__()
#         self.params = params
#         self.in_chns = self.params['in_chns']
#         self.ft_chns = self.params['feature_chns']
#         self.n_class = self.params['class_num']
#         self.bilinear = self.params['bilinear']
#         self.dropout = self.params['dropout']
#         assert (len(self.ft_chns) == 5)
#         self.in_conv = ConvBlock(
#             self.in_chns, self.ft_chns[0], self.dropout[0])
#         self.down1 = DownBlock(
#             self.ft_chns[0], self.ft_chns[1], self.dropout[1])
#         self.down2 = DownBlock(
#             self.ft_chns[1], self.ft_chns[2], self.dropout[2])
#         self.down3 = DownBlock(
#             self.ft_chns[2], self.ft_chns[3], self.dropout[3])
#         self.down4 = DownBlock(
#             self.ft_chns[3], self.ft_chns[4], self.dropout[4])
#
#     def forward(self, x):
#         x0 = self.in_conv(x)
#         x1 = self.down1(x0)
#         x2 = self.down2(x1)
#         x3 = self.down3(x2)
#         x4 = self.down4(x3)
#         return [x0, x1, x2, x3, x4]
#
#
# class Decoder(nn.Module):
#     def __init__(self, params):
#         super(Decoder, self).__init__()
#         self.params = params
#         self.in_chns = self.params['in_chns']
#         self.ft_chns = self.params['feature_chns']
#         self.n_class = self.params['class_num']
#         self.bilinear = self.params['bilinear']
#         assert (len(self.ft_chns) == 5)
#
#         self.up1 = UpBlock(
#             self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
#         self.up2 = UpBlock(
#             self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
#         self.up3 = UpBlock(
#             self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
#         self.up4 = UpBlock(
#             self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)
#
#         self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
#                                   kernel_size=3, padding=1)
#
#     def forward(self, feature):
#         x0 = feature[0]
#         x1 = feature[1]
#         x2 = feature[2]
#         x3 = feature[3]
#         x4 = feature[4]
#
#         x = self.up1(x4, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         x = self.up4(x, x0)
#         output = self.out_conv(x)
#         return output
#
#
# class Decoder_DS(nn.Module):
#     def __init__(self, params):
#         super(Decoder_DS, self).__init__()
#         self.params = params
#         self.in_chns = self.params['in_chns']
#         self.ft_chns = self.params['feature_chns']
#         self.n_class = self.params['class_num']
#         self.bilinear = self.params['bilinear']
#         assert (len(self.ft_chns) == 5)
#
#         self.up1 = UpBlock(
#             self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
#         self.up2 = UpBlock(
#             self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
#         self.up3 = UpBlock(
#             self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
#         self.up4 = UpBlock(
#             self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)
#
#         self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
#                                   kernel_size=3, padding=1)
#         self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
#                                       kernel_size=3, padding=1)
#         self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
#                                       kernel_size=3, padding=1)
#         self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
#                                       kernel_size=3, padding=1)
#         self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
#                                       kernel_size=3, padding=1)
#
#     def forward(self, feature, shape):
#         x0 = feature[0]
#         x1 = feature[1]
#         x2 = feature[2]
#         x3 = feature[3]
#         x4 = feature[4]
#         x = self.up1(x4, x3)
#         dp3_out_seg = self.out_conv_dp3(x)
#         dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)
#
#         x = self.up2(x, x2)
#         dp2_out_seg = self.out_conv_dp2(x)
#         dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)
#
#         x = self.up3(x, x1)
#         dp1_out_seg = self.out_conv_dp1(x)
#         dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)
#
#         x = self.up4(x, x0)
#         dp0_out_seg = self.out_conv(x)
#         return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg
#
#
# class Decoder_URPC(nn.Module):
#     def __init__(self, params):
#         super(Decoder_URPC, self).__init__()
#         self.params = params
#         self.in_chns = self.params['in_chns']
#         self.ft_chns = self.params['feature_chns']
#         self.n_class = self.params['class_num']
#         self.bilinear = self.params['bilinear']
#         assert (len(self.ft_chns) == 5)
#
#         self.up1 = UpBlock(
#             self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
#         self.up2 = UpBlock(
#             self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
#         self.up3 = UpBlock(
#             self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
#         self.up4 = UpBlock(
#             self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)
#
#         self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
#                                   kernel_size=3, padding=1)
#         self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
#                                       kernel_size=3, padding=1)
#         self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
#                                       kernel_size=3, padding=1)
#         self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
#                                       kernel_size=3, padding=1)
#         self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
#                                       kernel_size=3, padding=1)
#         self.feature_noise = FeatureNoise()
#
#     def forward(self, feature, shape):
#         x0 = feature[0]
#         x1 = feature[1]
#         x2 = feature[2]
#         x3 = feature[3]
#         x4 = feature[4]
#         x = self.up1(x4, x3)
#         if self.training:
#             dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
#         else:
#             dp3_out_seg = self.out_conv_dp3(x)
#         dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)
#
#         x = self.up2(x, x2)
#         if self.training:
#             dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
#         else:
#             dp2_out_seg = self.out_conv_dp2(x)
#         dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)
#
#         x = self.up3(x, x1)
#         if self.training:
#             dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
#         else:
#             dp1_out_seg = self.out_conv_dp1(x)
#         dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)
#
#         x = self.up4(x, x0)
#         dp0_out_seg = self.out_conv(x)
#         return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg
#
#
# def Dropout(x, p=0.3):
#     x = torch.nn.functional.dropout(x, p)
#     return x
#
#
# def FeatureDropout(x):
#     attention = torch.mean(x, dim=1, keepdim=True)
#     max_val, _ = torch.max(attention.view(
#         x.size(0), -1), dim=1, keepdim=True)
#     threshold = max_val * np.random.uniform(0.7, 0.9)
#     threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
#     drop_mask = (attention < threshold).float()
#     x = x.mul(drop_mask)
#     return x
#
#
# class FeatureNoise(nn.Module):
#     def __init__(self, uniform_range=0.3):
#         super(FeatureNoise, self).__init__()
#         self.uni_dist = Uniform(-uniform_range, uniform_range)
#
#     def feature_based_noise(self, x):
#         noise_vector = self.uni_dist.sample(
#             x.shape[1:]).to(x.device).unsqueeze(0)
#         x_noise = x.mul(noise_vector) + x
#         return x_noise
#
#     def forward(self, x):
#         x = self.feature_based_noise(x)
#         return x
#
#
# class UNet(nn.Module):
#     def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
#         super(UNet, self).__init__()
#         self.name = name
#         params = {'in_chns': in_channels,
#                   'feature_chns': [16, 32, 64, 128, 256],
#                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
#                   'class_num': num_classes,
#                   'bilinear': False,
#                   'acti_func': 'relu'}
#
#         self.encoder = Encoder(params)
#         self.decoder = Decoder(params)
#
#     def forward(self, x):
#         feature = self.encoder(x)
#         output = self.decoder(feature)
#         return output