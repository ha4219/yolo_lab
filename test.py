import torch
from models.yolo import Model
from utils.datasets import LoadImageAndLabels

c = Model()
d = LoadImageAndLabels('/home/oem/lab/jdongha/docker/datasets/coco128')

x = torch.randn((10, 3, 448, 448))

print(c(x).shape)