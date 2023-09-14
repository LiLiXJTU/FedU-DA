import math
from functools import partial
import torch.nn as nn
import torch
from torch.nn import LayerNorm, Conv2d, Dropout, Linear, Softmax
from torch.nn.modules.utils import _pair
import copy
import local.Config as config
import torch.nn.functional as F
config_vit = config.get_CTranS_config()
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

class Transformer(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels, embed_dim,vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, patchsize, img_size, in_channels, embed_dim)
        self.encoder = Encoder(config, vis,embed_dim,img_size)

    def forward(self, input_ids):
        embedding_output= self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, patchsize, img_size, in_channels,out_channels):
        super().__init__()
        # img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        # patch的数量
        n_patches = (img_size // patchsize) * (img_size // patchsize)

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, out_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        #(b,384,14,14)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        #(b,196,384)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)d
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
class Encoder(nn.Module):
        def __init__(self, config, vis, embed_dim,h):
            super(Encoder, self).__init__()
            self.vis = vis
            self.layer = nn.ModuleList()
            self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
            for _ in range(config.transformer["num_layers"]):
                layer = Block(config, vis, embed_dim,h)
                self.layer.append(copy.deepcopy(layer))

        def forward(self, hidden_states):
            attn_weights = []
            for layer_block in self.layer:
                hidden_states, weights = layer_block(hidden_states)
                if self.vis:
                    attn_weights.append(weights)
            encoded = self.encoder_norm(hidden_states)
            return encoded
class Block(nn.Module):
    def __init__(self, config, vis,embed_dim,h):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # self.ffn = Mlp(config,embed_dim,embed_dim*4)
        self.ffn = SE_MLP(embed_dim,h,h,6)
        self.attn = Attention(config, vis,embed_dim)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        # se = self.se(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    # def load_from(self, weights, n_block):
    #     ROOT = f"Transformer/encoderblock_{n_block}"
    #     with torch.no_grad():
    #         query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #         key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #         value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #         out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
    #
    #         query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
    #         key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
    #         value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
    #         out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
    #
    #         self.attn.query.weight.copy_(query_weight)
    #         self.attn.key.weight.copy_(key_weight)
    #         self.attn.value.weight.copy_(value_weight)
    #         self.attn.out.weight.copy_(out_weight)
    #         self.attn.query.bias.copy_(query_bias)
    #         self.attn.key.bias.copy_(key_bias)
    #         self.attn.value.bias.copy_(value_bias)
    #         self.attn.out.bias.copy_(out_bias)
    #
    #         mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
    #         mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
    #         mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
    #         mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
    #
    #         self.ffn.fc1.weight.copy_(mlp_weight_0)
    #         self.ffn.fc2.weight.copy_(mlp_weight_1)
    #         self.ffn.fc1.bias.copy_(mlp_bias_0)
    #         self.ffn.fc2.bias.copy_(mlp_bias_1)
    #
    #         self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
    #         self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
    #         self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
    #         self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
class Attention(nn.Module):
    def __init__(self, config, vis, embed_dim):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        #4个头
        #self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.attention_head_size = int(embed_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        #config.hidden_size=64
        # self.query = Linear(config.hidden_size, self.all_head_size)
        # self.key = Linear(config.hidden_size, self.all_head_size)
        # self.value = Linear(config.hidden_size, self.all_head_size)
        self.query = Linear(embed_dim, self.all_head_size)
        self.key = Linear(embed_dim, self.all_head_size)
        self.value = Linear(embed_dim, self.all_head_size)

        # self.out = Linear(config.hidden_size, config.hidden_size)
        self.out = Linear(embed_dim, embed_dim)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
class SE_MLP(nn.Module):
    def __init__(self, ch_in, h,w,reduction=6):
        super(SE_MLP, self).__init__()
        self.h = h
        self.w = w
        self.MAX_pool = nn.AdaptiveMaxPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, _, c = x.size()
        y=x.transpose(1, 2).reshape(b, c, self.h // 4, self.w // 4)
        y = self.MAX_pool(y).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1).transpose(1, 2)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上
class Mlp(nn.Module):
    def __init__(self, config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        #196->784
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class transtoconv(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes,H,W, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(transtoconv, self).__init__()
        self.H=H
        self.W=W
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        B, _, C = x.shape
        # [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x.transpose(1, 2).reshape(B, C, self.H//4, self.W//4)
        x_r = self.act(self.bn(self.conv_project(x_r))) #(b,64,14,14)

        return F.interpolate(x_r, size=(self.H, self.W)) #(b,64,56,56)
class UFusionNet(nn.Module):
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
        self.embed_dim =196
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
        # self.trans1 = Transformer(config_vit, patchsize=4, img_size=224, in_channels=32, embed_dim=self.embed_dim,vis=False)
        # self.transtoconv1 = transtoconv(inplanes=self.embed_dim, outplanes=n_channels,H=224,W=224)
        self.trans2 = Transformer(config_vit, patchsize=4, img_size=112, in_channels=64, embed_dim=self.embed_dim, vis=False)
        self.transtoconv2 = transtoconv(inplanes=self.embed_dim, outplanes=n_channels*2, H=112, W=112)
        self.trans3 = Transformer(config_vit, patchsize=4, img_size=56, in_channels=128, embed_dim=self.embed_dim, vis=False)
        self.transtoconv3 = transtoconv(inplanes=self.embed_dim, outplanes=n_channels*4, H=56, W=56)
        self.trans4 = Transformer(config_vit, patchsize=4, img_size=28, in_channels=256, embed_dim=self.embed_dim, vis=False)
        self.transtoconv4 = transtoconv(inplanes=self.embed_dim, outplanes=n_channels*8, H=28, W=28)

        if num_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        #(B,3136,384)
        # trans_x1=self.trans1(x1)
        # cnn_x1 = self.transtoconv1(trans_x1)
        # fusion1=x1+cnn_x1
        x2 = self.down1(x1)
        trans_x2=self.trans2(x2)
        cnn_x2 = self.transtoconv2(trans_x2)
        fusion2=x2+cnn_x2
        x3 = self.down2(x2)
        trans_x3=self.trans3(x3)
        cnn_x3 = self.transtoconv3(trans_x3)
        fusion3=x3+cnn_x3
        x4 = self.down3(x3)
        trans_x4=self.trans4(x4)
        cnn_x4 = self.transtoconv4(trans_x4)
        fusion4=x4+cnn_x4
        x5 = self.down4(x4)
        x = self.up4(x5, fusion4)
        x = self.up3(x, fusion3)
        x = self.up2(x, fusion2)
        x = self.up1(x, x1)
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print("111")
        else:
            logits = self.outc(x)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2,3,224,224)).cuda()
    model = UFusionNet('UFusionNet',3,2).cuda()
    param1 = sum([param.nelement() for param in model.parameters()])
    y = model(x)
    print(model)
    total = 0
    for name, param in model.named_parameters():
        mul = 1
        for size in param.shape:
            mul = size * mul # 统计每层参数个数
            print(name,mul)
        total += mul
        print('1',total)
    print('2',total)
    print('Output shape:',y.shape)
    print('UNet totoal parameters: %.2fM (%d)'%(param1/1e6,param1))