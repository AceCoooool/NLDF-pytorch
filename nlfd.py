import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

base = {'352': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
extra = {'352': [2, 7, 14, 21, 28]}


# vgg16
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


class ConvConstract(nn.Module):
    def __init__(self, in_channel):
        super(ConvConstract, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=3, padding=1)
        self.cons1 = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x2 = self.cons1(x)
        return x, x - x2


# extra part
def extra_layer(vgg, cfg):
    feat_layers, pool_layers = [], []
    for k, v in enumerate(cfg):
        feat_layers += [ConvConstract(vgg[v].out_channels)]
        if k == 0:
            pool_layers += [nn.Conv2d(128 * (6 - k), 128 * (5 - k), 1)]
        else:
            # TODO: change this to sampling
            pool_layers += [nn.ConvTranspose2d(128 * (6 - k), 128 * (5 - k), 3, 2, 1, 1)]
    return vgg, feat_layers, pool_layers


class NLFD(nn.Module):
    def __init__(self, base, feat_layers, pool_layers):
        super(NLFD, self).__init__()
        self.pos = [4, 9, 16, 23, 30]
        self.base = nn.ModuleList(base)
        self.feat = nn.ModuleList(feat_layers)
        self.pool = nn.ModuleList(pool_layers)
        self.glob = nn.Sequential(nn.Conv2d(512, 128, 5), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 5),
                                  nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3))
        self.conv_g = nn.Conv2d(128, 1, 1)
        self.conv_l = nn.Conv2d(640, 1, 1)

    def forward(self, x, label=None):
        sources, num = list(), 0
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.pos:
                sources.append(self.feat[num](x))
                num = num + 1
        for k in range(4, -1, -1):
            if k == 4:
                out = F.relu(self.pool[k](torch.cat([sources[k][0], sources[k][1]], dim=1)), inplace=True)
            else:
                out = self.pool[k](torch.cat([sources[k][0], sources[k][1], out], dim=1)) if k == 0 else F.relu(
                    self.pool[k](torch.cat([sources[k][0], sources[k][1], out], dim=1)), inplace=True)

        score = self.conv_g(self.glob(x)) + self.conv_l(out)
        prob = F.sigmoid(score)
        return prob


def build_model():
    return NLFD(*extra_layer(vgg(base['352'], 3), extra['352']))


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

