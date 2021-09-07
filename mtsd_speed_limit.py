'''
End-to-end speed limit detection
Dataset: https://btsd.ethz.ch/shareddata/
Download and unpack following files

* https://btsd.ethz.ch/shareddata/BelgiumTS/Annotations/camera00.tar
* https://btsd.ethz.ch/shareddata/BelgiumTS/Annotations/camera01.tar
* https://btsd.ethz.ch/shareddata/BelgiumTS/BelgiumTSD_annotations.zip

into ./data/BelgiumTSD directory
'''

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import json

from file_list import MTSDDataset
from visualize_example import load_annotation, visualize_gt

plt.ion()  # interactive mode

# %%
"""
Data loader for original Mapillary MTSD dataset with two classes: 1 for an image with a speed limit and 0 without
"""

data_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_datasets = {
    'train':
    MTSDDataset(root='../datasets/mtsd_v2_fully_annotated/',
                flist='splits/train.txt',
                annotations_dir='annotations/',
                images_dir='images/train/',
                transform=data_transform),
    'val':
    MTSDDataset(root='../datasets/mtsd_v2_fully_annotated/',
                flist='splits/val.txt',
                annotations_dir='annotations/',
                images_dir='images/val/',
                transform=data_transform)
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#  class_names = image_datasets['train'].classes
class_names = ['no_limit', 'limit']

# %%

inputs, classes = next(iter(dataloaders['train']))

image_key = inputs

# load the annotation json
anno = load_annotation(image_key)

# visualize traffic sign boxes on the image
vis_img = visualize_gt(image_key, anno)
vis_img.show()
