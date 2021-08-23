# Process MTSD dataset to use with YOLOv5

import os
import sys
import errno
import shutil
import json
import random

LABEL_GROUPS = ['regulatory--maximum-speed-limit']
OTHER_SIGN_LABEL = None     # set to None to ignore labels not in LABEL_GROUPS
BACKGROUND_IMAGES_RATIO = 0.1
DATASET_PATH = '../datasets/mtsd_v2_fully_annotated'
NEW_DATASET_PATH = '../datasets/mtsd_speed_limits'

SETS = ['train', 'val']


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


labels_path = f'{DATASET_PATH}/labels'
if os.path.isdir(labels_path):
    print("Removing directory with labels")
    shutil.rmtree(labels_path)

try:
    os.makedirs(f'{DATASET_PATH}/labels')
    for s in SETS:
        os.makedirs(f'{DATASET_PATH}/labels/{s}')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

labels = {}
image_keys_with_labels = {s: [] for s in SETS}
image_keys_background = {s: [] for s in SETS}

for s in SETS:
    file_name = f'{DATASET_PATH}/splits/{s}.txt'
    file_size = os.path.getsize(file_name)
    read_count = 0
    count = 0
    with open(file_name) as f:
        for image_key in f:
            read_count += len(image_key)
            if not count % 100 or read_count == file_size:
                progress(read_count, file_size, f'Processing annotations for {s}')
            count += 1

            image_key = image_key.rstrip('\n')
            annotation_path = f'{DATASET_PATH}/annotations/{image_key}.json'
            annotation_new_path = f'{DATASET_PATH}/labels/{s}/{image_key}.json'
            shutil.copy(annotation_path, annotation_new_path)
            with open(annotation_new_path) as annotation_file:
                annotation_json = json.load(annotation_file)

            label_file_path = os.path.splitext(annotation_new_path)[0] + '.txt'
            label_file_content = ''
            for obj in annotation_json.get('objects'):
                label = obj['label']

                # if the label does not belong to one of the groups defined in LABEL_GROUPS,
                # rename it to OTHER_SIGN_LABEL,
                # leave original label if label groups are not defined
                if len(LABEL_GROUPS):
                    new_label = OTHER_SIGN_LABEL
                    for l_g in LABEL_GROUPS:
                        if label.startswith(l_g):
                            new_label = label
                            break
                    label = new_label

                if label is not None:
                    if label not in labels:
                        labels[label] = len(labels)
                    x_center = (obj['bbox']['xmin'] +
                                (obj['bbox']['xmax'] - obj['bbox']['xmin']) / 2) / annotation_json['width']
                    y_center = (obj['bbox']['ymin'] +
                                (obj['bbox']['ymax'] - obj['bbox']['ymin']) / 2) / annotation_json['height']
                    width = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / annotation_json['width']
                    height = (obj['bbox']['ymax'] - obj['bbox']['ymin']) / annotation_json['height']
                    label_str = f'{labels[label]} {x_center} {y_center} {width} {height}'
                    label_file_content += f'{label_str}\n'

            # if none of the objects had label in the LABEL_GROUPS, label_file_content will be an empty sting
            if label_file_content:
                with open(label_file_path, 'x') as label_file:
                    label_file.write(label_file_content)
                    image_keys_with_labels[s].append(image_key)
            elif OTHER_SIGN_LABEL is None:
                image_keys_background[s].append(image_key)

            os.remove(annotation_new_path)
    print()


print("Finished creating labels for yolov5")

if os.path.isdir(NEW_DATASET_PATH) and image_keys_background:
    print(f"Removing {NEW_DATASET_PATH} directory")
    shutil.rmtree(NEW_DATASET_PATH)
try:
    os.makedirs(NEW_DATASET_PATH)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

if OTHER_SIGN_LABEL is None and BACKGROUND_IMAGES_RATIO > 0:
    print("Creating new image dataset with background images")
    for s in SETS:
        labels_path = f'{DATASET_PATH}/labels/{s}'
        images_path = f'{DATASET_PATH}/images/{s}'
        new_images_path = f'{NEW_DATASET_PATH}/images/{s}'
        new_labels_path = f'{NEW_DATASET_PATH}/labels/{s}'

        try:
            os.makedirs(new_images_path)
            os.makedirs(new_labels_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        total_files = len(image_keys_with_labels[s])
        count = 0
        for image_key in image_keys_with_labels[s]:
            shutil.copy(f'{images_path}/{image_key}.jpg', f'{new_images_path}/{image_key}.jpg')
            shutil.copy(f'{labels_path}/{image_key}.txt', f'{new_labels_path}/{image_key}.txt')
            if not count % 100 or count == total_files:
                progress(count, total_files, f"Copying labeled images for {s}")
            count += 1
        print()

        total_files = len(image_keys_background[s])
        count = 0
        background_img_prob = len(image_keys_with_labels[s]) * BACKGROUND_IMAGES_RATIO / total_files
        for image_key in image_keys_background[s]:
            if random.choices([0, 1], [1 - background_img_prob, background_img_prob])[0]:
                shutil.copy(f'{images_path}/{image_key}.jpg', f'{new_images_path}/{image_key}.jpg')
            if not count % 100 or count == total_files:
                progress(count, total_files, f"Copying background images for {s}")
            count += 1
        print()
    DATASET_PATH = NEW_DATASET_PATH

dataset_yaml_path = f'{DATASET_PATH}/mtsd.yaml'
print(f"Creating {dataset_yaml_path} file")
with open(dataset_yaml_path, 'w+') as dataset_file:
    dataset_file.write(f'path: {DATASET_PATH}\n')
    dataset_file.write('train: images/train\n')
    dataset_file.write('val: images/val\n')
    dataset_file.write(f'nc: {len(labels)}\n')
    dataset_file.write(f'names: {list(labels.keys())}\n')
