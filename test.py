import os
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import numpy as np
from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.general import LOGGER, check_dataset, check_yaml
from torch.optim import SGD, Adam, AdamW, lr_scheduler


WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))

model = Model()
hyp = check_yaml('data/hyps/hyp.yaml')
with open(hyp, errors='ignore') as f:
    hyp = yaml.safe_load(f)
x = torch.randn((10, 3, 448, 448))

# gs = max(int(model.stride.max()), 32)  # grid size (max stride)
gs = 7
single_cls = 80

imgsz = 448
batch_size = 64

data = 'data/coco.yaml'
data_dict = check_dataset(data)
train_path, val_path = data_dict['train'], data_dict['val']
g = [], [], []  # optimizer parameter groups
bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
for v in model.modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
        g[2].append(v.bias)
    if isinstance(v, bn):  # weight (no decay)
        g[1].append(v.weight)
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
        g[0].append(v.weight)

# Trainloader
train_loader, dataset = create_dataloader(train_path,
                                            imgsz,
                                            batch_size // WORLD_SIZE,
                                            gs,
                                            single_cls,
                                            hyp=hyp,
                                            augment=True,
                                            # cache=None if opt.cache == 'val' else opt.cache,
                                            # rect=opt.rect,
                                            rank=LOCAL_RANK,
                                            workers=4,
                                            # image_weights=opt.image_weights,
                                            # quad=opt.quad,
                                            # prefix=colorstr('train: '),
                                            shuffle=True)
val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                    #    cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=4,
                                       pad=0.5,
                                    #    prefix=colorstr('val: ')
                                       )[0]
                               

optimizer = Adam(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
device = 'cuda:0'
nbs = 64
nb = len(train_loader)
nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
# scaler = torch.cuda.amp.GradScaler(enabled=amp)
epochs = 50
# Scheduler
lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

for epoch in range(50):
    model.train()
    if RANK != -1:
        train_loader.sampler.set_epoch(epoch)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
    if RANK in {-1, 0}:
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    optimizer.zero_grad()

    for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

        # Warmup
        if ni <= nw:
            xi = [0, nw]  # x interp
            # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

        # Forward
        print(imgs.shape, targets)
        pred = model(imgs)  # forward
        print(pred)
        #     loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
        #     if RANK != -1:
        #         loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
        
        # scaler.scale(loss).backward()
