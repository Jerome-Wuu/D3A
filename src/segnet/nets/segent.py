# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : segent.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        idx = []

        x = self.encode1(x)
        x, id1 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id1)

        x = self.encode2(x)
        x, id2 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id2)

        x = self.encode3(x)
        x, id3 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id3)

        return x, idx


class Deocder(nn.Module):
    def __init__(self, out_channels):
        super(Deocder, self).__init__()

        self.decode1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decode2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

        )

    def forward(self, x, idx):
        """
        :param x: 经过卷积操作后的特征图
        :param idx: decode中每次最大池化时最大值的位置索引
        """

        x = F.max_unpool2d(x, idx[2], kernel_size=2, stride=2)
        x = self.decode1(x)

        x = F.max_unpool2d(x, idx[1], kernel_size=2, stride=2)
        x = self.decode2(x)

        x = F.max_unpool2d(x, idx[0], kernel_size=2, stride=2)
        x = self.decode3(x)

        return x


class ShapeChangeOut(nn.Module):
    def __init__(self, out_channels):
        super(ShapeChangeOut, self).__init__()
        self.out_channels = out_channels

    def forward(self, x):
        x = F.interpolate(x.unsqueeze(0), size=(self.out_channels, 84, 84), mode='nearest').squeeze(0)
        return x

class ShapeChangePre(nn.Module):
    def __init__(self):
        super(ShapeChangePre, self).__init__()

    def forward(self, x):
        x = F.interpolate(x.unsqueeze(0), size=(3, 128, 128), mode='nearest').squeeze(0)
        return x

class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        self.encoder = Encoder(in_channels=3)
        self.decoder = Deocder(out_channels=num_classes)
        self.shape_change_pre = ShapeChangePre()
        self.shape_change_out = ShapeChangeOut(out_channels=num_classes)


    def forward(self, x):

        x = self.shape_change_pre(x)
        x, idx = self.encoder(x)

        x = self.decoder(x, idx)
        x = self.shape_change_out(x)
        return x


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 512, 512)
    model = SegNet(num_classes=2)
    output = model(inputs)
    print(output.shape)