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

from file_list import ImageAnnotationfiles
from data.mtsd_v2_fully_annotated import visualize_example

plt.ion()  # interactive mode

# %%


def mtsd_annotations_reader(root, flist, annotations_dir, images_dir):
    imlist = []
    with open(os.path.join(root, flist), 'r') as rf:
        for line in rf.readlines():
            impath = os.path.join(root, images_dir, line.rstrip() + '.jpg')
            anpath = os.path.join(root, annotations_dir,
                                  line.rstrip() + '.json')
            try:
                if os.path.isfile(impath):
                    with open(anpath, 'r') as afile:
                        annotation = json.load(afile)
                        labels = [
                            item.get('label', None)
                            for item in annotation.get('objects', {})
                        ]
                        is_speed_limit = 0
                        for label in labels:
                            if label.startswith(
                                    'regulatory--maximum-speed-limit'):
                                is_speed_limit = 1
                                break
                        imlist.append((os.path.join(images_dir,
                                                    line.rstrip() + '.jpg'),
                                       is_speed_limit))
                else:
                    open(impath, 'r')
            except FileNotFoundError as e:
                print("File not found: " + e.filename)

    return imlist


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {
    'train':
    ImageAnnotationfiles(root='./data/mtsd_v2_fully_annotated/',
                         flist='splits/train.txt',
                         annotations_dir='annotations/',
                         images_dir='images/',
                         transform=data_transform,
                         annotations_reader=mtsd_annotations_reader),
    'val':
    ImageAnnotationfiles(root='./data/mtsd_v2_fully_annotated/',
                         flist='splits/val.txt',
                         annotations_dir='annotations/',
                         images_dir='images/',
                         transform=data_transform,
                         annotations_reader=mtsd_annotations_reader)
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x],
                                   batch_size=4,
                                   shuffle=True,
                                   num_workers=4)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#  class_names = image_datasets['train'].classes
class_names = ['no_limit', 'limit']

# %%

inputs, classes = next(iter(dataloaders['train']))

image_key = inputs

# load the annotation json
anno = visualize_example.load_annotation(image_key)

# visualize traffic sign boxes on the image
vis_img = visualize_example.visualize_gt(image_key, anno)
vis_img.show()
