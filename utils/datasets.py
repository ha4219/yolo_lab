import os

from pathlib import Path
from glob import glob
import torch
from torch.utils.data import DataLoader, Dataset, dataloader, distributed

IMG_FORMATS = ['png', 'jpg', 'jpeg']
LABEL_FORMATS = ['txt']

class LoadImageAndLabels(Dataset):
    """
        dataset module
    """

    def __init__(
        self,
        path,
        img_size=448,
        fs=7,
        nb=2,
        nc=80,
    ):
        self.img_size = img_size
        self.path = path

        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob(str(p/'*'/'*'/'*.*'), recursive=True)
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        except Exception as e:
            raise Exception(f'image file error')

        self.label_files = img2label_paths(self.im_files)

        print(self.im_files[-1], self.label_files[-1])

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]