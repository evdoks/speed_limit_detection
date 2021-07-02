from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy

from file_list import ImageFilelist


plt.ion()  # interactive mode

# %%


def btsd_flist_reader(flist):
    """
    flist format: [camera]/[image];[x1];[y1];[x2];[y2];[class id];[superclass id];
    """
    imdict = {}
    imlist = []
    speed_limit_class_id = '65'   # class id 65 is speed limit (superclass id 2)
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            if line[:2] in ['00', '01']:
                annotations = line.split(';')
                impath = annotations[0]
                imlabel = 1 if annotations[5] == speed_limit_class_id else 0
                if impath not in imdict or imdict[impath] != 1:
                    imdict[impath] = imlabel
    imlist = list(imdict.items())

    return imlist

# %%


data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_loader = torch.utils.data.DataLoader(
    ImageFilelist(root="./data/BelgiumTSD/",
                  flist="./data/BelgiumTSD/BelgiumTSD_annotations/BTSD_training_GTclear.txt",
                  transform=data_transform,
                  flist_reader=btsd_flist_reader),
    batch_size=8, shuffle=True,
    num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    ImageFilelist(root="./data/BelgiumTSD/",
                  flist="./data/BelgiumTSD/BelgiumTSD_annotations/BTSD_testing_GTclear.txt",
                  transform=data_transform,
                  flist_reader=btsd_flist_reader),
    batch_size=8, shuffle=True,
    num_workers=4, pin_memory=True)


# %%

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


train_loader_iter = iter(train_loader)

# %%
# Get a batch of training data
inputs, classes = next(train_loader_iter)

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

class_names = ['no limit', 'limit']
imshow(out, title=[class_names[x] for x in classes])
