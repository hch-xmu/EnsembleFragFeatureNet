# -*- coding: utf-8 -*-
from __future__ import division

""" 
Creates a ResNeXt Model

"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, kernel_s, padding_s, stride, cardinality, base_width, widen_factor):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            kernel_s: Define convolution kernel size (h, w)
            padding_s: Define padding (if any) (l, r, t, b)
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * base_width) # out_channels = 2^n * widen_factor * base_width
        D = cardinality * int(base_width * width_ratio) # should be double out_channels when C = 2*widen_factor
        
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.padd = nn.ZeroPad2d(padding_s) #asymmetric padding dependent on even kernel 
        self.conv_conv = nn.Conv2d(D, D, kernel_size=kernel_s, stride=stride, padding=0, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(self.padd.forward(bottleneck)) #added padding
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class FeatureResNeXt(nn.Module):

    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(FeatureResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // (3 * 7) # / no_layers per bottleneck * no_stages
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [self.base_width * self.widen_factor, 2* self.base_width * self.widen_factor, 4* self.base_width * self.widen_factor, 8 * self.base_width * self.widen_factor, 16 * self.base_width * self.widen_factor, 32 * self.base_width * self.widen_factor]         #[32, 64, 128, 256, 512, 1024]        
        
        self.padds = nn.ZeroPad2d((40,39,1,0)) #asymmetric padding dependent on even kernel 
        self.conv_1_3x3 = nn.Conv2d(1, 8, (2,80), 1, 0, bias=False) #in, out, kernel, stride, padding (l,r,t,b)
        self.bn_1 = nn.BatchNorm2d(8)
        
        #(name, in, out, kernel_s, padding_s, pool_stride)
        self.stage_A = self.block('stage_A', 8, 16, (2,80), (40,39,1,0) ,1)
        self.stage_B = self.block('stage_B', 16, 32, (2,80), (40,39,1,0) ,1)
        self.stage_C = self.block('stage_C', 32, 64, (2,80), (40,39,1,0) ,1)
        self.stage_D = self.block('stage_D', 64, 128, (2,80), (40,39,1,0) ,1)

        self.stage_1 = self.block('stage_1', 128, 256, (2,20), (10,9,1,0), (1,2))
        self.stage_2 = self.block('stage_2', 256, 512, (2,10), (5,4,1,0), (1,2))
        self.stage_3 = self.block('stage_3', 512, 1024, 3, 1, (2,2))
        
        self.classifier = nn.Linear(1024, nlabels)
        init.kaiming_normal_(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, kernel_s, padding_s, pool_stride):
        """ 
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, kernel_s, padding_s, pool_stride, self.cardinality, self.base_width, self.widen_factor))
            else:
                block.add_module(name_, ResNeXtBottleneck(out_channels, out_channels, kernel_s, padding_s, 1, self.cardinality, self.base_width, self.widen_factor))
        return block

    def forward(self, x): 
        x = self.conv_1_3x3.forward(self.padds.forward(x))  #added padding ([32, 1, 6, 80]) -> ([32, 8, 6, 80])
        x = F.relu(self.bn_1.forward(x), inplace=True)      # ([32, 1, 6, 80]) -> ([32, 8, 6, 80])
        x = self.stage_A.forward(x)                        # ([32, 8, 6, 80]) -> ([32, 16, 6, 80])
        x = self.stage_B.forward(x)                        # ([32, 16, 6, 80]) -> ([32, 32, 6, 80])
        x = self.stage_C.forward(x)                        # ([32, 32, 6, 80]) -> ([32, 64, 6, 80])
        x = self.stage_D.forward(x)                        # ([32, 64, 6, 80]) -> ([32, 128, 6, 80])
        x = self.stage_1.forward(x)                        # ([32, 128, 6, 80]) -> ([32, 256, 6, 40])
        x = self.stage_2.forward(x)                        # ([32, 256, 6, 40]) -> ([32, 512, 6, 20])
        x = self.stage_3.forward(x)                        # ([32, 512, 6, 20]) -> ([32, 1024, 3, 10])
        x = F.avg_pool2d(x, (3,10), 1)                      # ([32, 1024, 3, 10]) -> ([32, 1024, 1, 1])
        x = x.view(-1, 1024)                                 # ([32, 1024, 1, 1]) -> ([32, 1024])
        a=self.classifier(x)                                # ([32, 1024]) -> ([32, 20])
        return a
