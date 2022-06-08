# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
import yaml

import torch
import torch.nn as nn

from models.common import *


class Detect(nn.Module):
    def __init__(self, fs=7, nb=2, nc=80):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False), # is it okay to use Dropout with BatchNorm?
            nn.Linear(4096, fs * fs * (5 * nb + nc)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)
    
    def forward_fuse(self, x):
        return self.fc(x)

class Model(nn.Module):
    def __init__(self, cfg='models/shape/v1/yolov1.yaml', ch=3):
        super().__init__()
        with open(cfg, encoding='ascii', errors='ignore') as f:
            self.yaml = yaml.safe_load(f)  # model dict

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.fs, self.nb, self.nc = self.yaml.get('fs', 7), self.yaml.get('nb', 2), self.yaml.get('nc', 80)
        self.model = parse_model(deepcopy(self.yaml), [ch])
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.fs, self.fs, 5 * self.nb + self.nc)
        return x

    def forward_fuse(self, x):
        x = self.model(x)
        x = x.view(-1, self.fs, self.fs, 5 * self.nb + self.nc)
        return x 
    
def parse_model(d, ch):
    """
        마지막 fc는 여기서 작성
    """
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # from, number, module, args
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        if m is Conv_v1:
            c1, c2 = ch[f], args[0]

            args = [c1, c2, *args[1:]]
        elif m is Maxpool_v1:
            c1, c2 = ch[f], ch[f]
        elif m is Detect:
            args = [d[item] for item in args]
        m_ = nn.Sequential(*(m(*args) for _ in range(n)))
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers)


