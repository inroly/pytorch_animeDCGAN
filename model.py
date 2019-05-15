# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 23:41:10 2018

@author: dlf43
"""

import torch
import torch.nn as nn


class NetG(nn.Module):
    """ 生成器 """
    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = opt.ngf  # 生成器的feature map数
        
        self.main = nn.Sequential(
            # 转置卷积操作
            # ConvTranspose2d(in_channels, out_channels, kernel_size, 2, 1, bias=False),
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # 输出 (ngf * 8) * 4 * 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # 输出 (ngf * 4) * 8 * 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # 输出 (ngf * 2) * 16 * 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # 输出 ngf * 32 * 32
            
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 归一化到 (-1, 1)
            # 输出 3 * 96 * 96
        )
    
    def forward(self, input):
        return self.main(input)


class NetD(nn.Module):
    """ 判别器 """
    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf

        self.main = nn.Sequential(
            # 输入 3 * 96 * 96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 输出 ndf * 32 * 32

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*2) * 16 * 16

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*4) * 8 * 8

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*8) * 4 * 4

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出概率
        )
    
    def forward(self, input):
        return self.main(input).view(-1)
