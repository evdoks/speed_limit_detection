# Process MTSD dataset to use with YOLOv5

import os
import errno
import shutil
import json

LABEL_GROUPS = ['regulatory--maximum-speed-limit']
OTHER_SIGN_LABEL = 'other-sign'

DATASET_PATH = './mtsd_v2_fully_annotated'
SETS = ['train', 'val']

labels_path = f'{DATASET_PATH}/labels'
if os.path.isdir(labels_path):
    print("Removing directory with labels")
    shutil.rmtree(labels_path)

try:
    os.makedirs(f'{DATASET_PATH}/labels')
    for d in SETS:
        os.makedirs(f'{DATASET_PATH}/labels/{d}')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

labels = {}
for d in SETS:
    print(f'Processing {d}')
    with open(f'{DATASET_PATH}/splits/{d}.txt') as f:
        for line in f:
            line = line.rstrip('\n')
            annotation_path = f'{DATASET_PATH}/annotations/{line}.json'
            annotation_new_path = f'{DATASET_PATH}/labels/{d}/{line}.json'
            try:
                shutil.copy(annotation_path, annotation_new_path)
                with open(annotation_new_path) as annotation_file:
                    annotation_json = json.load(annotation_file)
                label_file_path = os.path.splitext(annotation_new_path)[0] + '.txt'
                with open(label_file_path, 'x') as label_file:
                    for obj in annotation_json.get('objects'):
                        label = obj['label']

                        # if the label belong to one of the groups defined in LABEL_GROUPS,
                        # replace the label with a group name, rename to OTHER_SIGN_LABEL otherwise,
                        # leave original label if label groups are not defined
                        if len(LABEL_GROUPS) != 0:
                            for l_g in LABEL_GROUPS:
                                if label.startswith(l_g):
                                    break
                                label = OTHER_SIGN_LABEL

                        if label not in labels:
                            labels[label] = len(labels)
                        x_center = (obj['bbox']['xmin'] +
                                    (obj['bbox']['xmax'] - obj['bbox']['xmin']) / 2) / annotation_json['width']
                        y_center = (obj['bbox']['ymin'] +
                                    (obj['bbox']['ymax'] - obj['bbox']['ymin']) / 2) / annotation_json['height']
                        width = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / annotation_json['width']
                        height = (obj['bbox']['ymax'] - obj['bbox']['ymin']) / annotation_json['height']
                        label_str = f'{labels[label]} {x_center} {y_center} {width} {height}'
                        label_file.write(f'{label_str}\n')
                os.remove(annotation_new_path)
            except FileNotFoundError as e:
                print(f'File not found: {e.filename}')

dataset_yaml_path = './mtsd.yaml'
with open(dataset_yaml_path, 'w+') as dataset_file:
    dataset_file.write(f'path: {DATASET_PATH}\n')
    dataset_file.write('train: images/train\n')
    dataset_file.write('val: images/val\n')
    dataset_file.write(f'nc: {len(labels)}\n')
    dataset_file.write(f'names: {list(labels.keys())}\n')
