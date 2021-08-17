# convert 'other-sign' images into background images and adjust their number according to defined ratio

import sys
import os
import errno
import yaml
import shutil
import random

LABEL_GROUPS = ['regulatory--maximum-speed-limit']
BACKGROUND_LABEL = 'other-sign'

DATASET_PATH = './mtsd_v2_fully_annotated'
NEW_DATASET_PATH = './mtsd_speed_limits'
IMAGES_DIR = 'images'
LABELS_DIR = 'labels'
#  NEW_IMAGES_DIR = 'speed_limit_images'
#  NEW_LABELS_DIR = 'speed_limit_labels'
SETS = ['train', 'val']
BACKGROUND_IMAGES_RATIO = 0.1
DATASET_YAML_PATH = 'mtsd.yaml'
SETS = ['train', 'val']

new_labels = {}


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


with open(DATASET_YAML_PATH, 'r') as dataset_file:
    dataset = yaml.safe_load(dataset_file)

if os.path.isdir(NEW_DATASET_PATH):
    print(f"Removing {NEW_DATASET_PATH} directory")
    shutil.rmtree(NEW_DATASET_PATH)
try:
    os.makedirs(NEW_DATASET_PATH)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for s in SETS:
    labels_path = f'{DATASET_PATH}/{LABELS_DIR}/{s}'
    images_path = f'{DATASET_PATH}/{IMAGES_DIR}/{s}'
    new_images_path = f'{NEW_DATASET_PATH}/{IMAGES_DIR}/{s}'
    new_labels_path = f'{NEW_DATASET_PATH}/{LABELS_DIR}/{s}'

    try:
        os.makedirs(f'{new_images_path}')
        os.makedirs(f'{new_labels_path}')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    print(f"Copying images with {LABEL_GROUPS} prefixes")
    total_files = len(os.listdir(labels_path))
    counter = 0
    for label_filename in os.listdir(labels_path):
        with open(f'{labels_path}/{label_filename}') as label_file:
            for line in label_file:
                label_idx = int(line.split()[0])
                label = dataset['names'][label_idx]

                # get first element of labels list of return None if the list is empty
                new_label = next(iter([label_prefix for label_prefix in LABEL_GROUPS
                                       if label.startswith(label_prefix)]), None)
                if new_label:
                    image_filename = label_filename.replace(LABELS_DIR, IMAGES_DIR).replace('.txt', '.jpg')
                    shutil.copy(f'{images_path}/{image_filename}', f'{new_images_path}/{image_filename}')
                    with open(f'{new_labels_path}/{label_filename}', 'x') as new_label_file:
                        if new_label not in new_labels:
                            new_labels[new_label] = len(new_labels)
                        line = line.split()
                        line[0] = str(new_labels[new_label])
                        new_label_file.write(f'{line}\n')

    counter += 1
    if counter % 100:
        progress(counter, total_files, f"processing {labels_path}")

    print("Adding background images")
    new_img_count = len(os.listdir(new_images_path))
    background_img_count = len(os.listdir(images_path)) - new_img_count
    background_img_prob = new_img_count * BACKGROUND_IMAGES_RATIO / background_img_count

    total_files = len(os.listdir(labels_path))
    counter = 0

    for label_filename in os.listdir(labels_path):
        with open(f'{labels_path}/{label_filename}') as label_file:
            for line in label_file:
                label_idx = int(line.split()[0])
                label = dataset['names'][label_idx]
                if not len([label_prefix for label_prefix in LABEL_GROUPS if label.startswith(label_prefix)
                            ]) and random.choices([0, 1], [1 - background_img_prob, background_img_prob])[0]:
                    image_filename = label_filename.replace(LABELS_DIR, IMAGES_DIR).replace('.txt', '.jpg')
                    # not labels are required for background images
                    #  shutil.copy(f'{labels_path}/{label_filename}',
                    #              f'{labels_path}/{label_filename}'.replace(LABELS_DIR, NEW_LABELS_DIR))
                    shutil.copy(f'{images_path}/{image_filename}', f'{new_images_path}/{image_filename}')
        counter += 1
        if counter % 100:
            progress(counter, total_files, f"processing {labels_path}")

dataset_yaml_path = f'{NEW_DATASET_PATH}/mtsd_speed_limit.yaml'
with open(dataset_yaml_path, 'w+') as dataset_file:
    dataset_file.write(f'path: {DATASET_PATH}\n')
    dataset_file.write(f'train: {IMAGES_DIR}/train\n')
    dataset_file.write(f'val: {IMAGES_DIR}/val\n')
    dataset_file.write(f'nc: {len(new_labels)}\n')
    dataset_file.write(f'names: {list(new_labels.keys())}\n')
