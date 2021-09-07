import os
import yaml
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob
import errno


DATASET_PATH = '../../datasets/mtsd_speed_limits'
NEW_DATASET_PATH = '../../datasets/mtsd_speed_limits_sliced'

# specify path for a new tiled dataset
SETS = ['train', 'val']

# specify slice width=height
SLIZE_SIZE = 512

# consider only slices with bounding box with a larger width or height and containing at least a certain fraction of it
MAX_HIGHT_WIDTH_RATIO = 1.5
MIN_INTERCECTION_RATIO = 0.5

# ratio of background images
BACKGROUND_IMAGES_RATIO = 0.5


def slice_dataset():
    if os.path.isdir(NEW_DATASET_PATH):
        print("Removing directory with tiles")
        shutil.rmtree(NEW_DATASET_PATH)
    os.makedirs(NEW_DATASET_PATH)

    print("Copying yaml file")
    yaml_files = glob.glob(f'{DATASET_PATH}/*.yaml')
    if len(yaml_files):
        shutil.copy(yaml_files[0], f'{NEW_DATASET_PATH}')
        with open(f'{NEW_DATASET_PATH}/{yaml_files[0].split(os.sep)[-1]}', 'r+') as f:
            dataset_dict = yaml.safe_load(f)
            dataset_dict['path'] = NEW_DATASET_PATH
            f.seek(0)
            yaml.dump(dataset_dict, f, default_flow_style=False)
            f.truncate()

    for s in SETS:
        images_path = f'{DATASET_PATH}/images/{s}'
        new_images_path = f'{NEW_DATASET_PATH}/images/{s}'
        new_labels_path = f'{NEW_DATASET_PATH}/labels/{s}'

        try:
            os.makedirs(new_images_path)
            os.makedirs(new_labels_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # get all image names
        imnames = glob.glob(f'{images_path}/*.jpg')

        image_count = 0
        background_count = 0

        # tile all images in a loop
        for imname in imnames:
            try:
                labname = imname.replace('.jpg', '.txt').replace('images', 'labels')
                labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
            except FileNotFoundError:
                print(f"Skipping background image {imname}")

            im = Image.open(imname)
            imr = np.array(im, dtype=np.uint8)
            height = imr.shape[0]
            width = imr.shape[1]

            # we need to rescale coordinates from 0-1 to real image height and width
            labels[['x1', 'w']] = labels[['x1', 'w']] * width
            labels[['y1', 'h']] = labels[['y1', 'h']] * height

            boxes = []

            # convert bounding boxes to shapely polygons.
            # We need to invert Y and find polygon vertices from center points
            for row in labels.iterrows():
                x1 = row[1]['x1'] - row[1]['w'] / 2
                y1 = (height - row[1]['y1']) - row[1]['h'] / 2
                x2 = row[1]['x1'] + row[1]['w'] / 2
                y2 = (height - row[1]['y1']) + row[1]['h'] / 2

                boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))

            counter = 0
            print('Image:', imname)
            # create tiles and find intersection with bounding boxes for each tile
            for i in range((height // SLIZE_SIZE)):
                for j in range((width // SLIZE_SIZE)):
                    x1 = j * SLIZE_SIZE
                    y1 = height - (i * SLIZE_SIZE)
                    x2 = ((j + 1) * SLIZE_SIZE) - 1
                    y2 = (height - (i + 1) * SLIZE_SIZE) + 1

                    pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                    imsaved = False
                    slice_labels = []

                    for box in boxes:
                        if pol.intersects(box[1]):
                            inter = pol.intersection(box[1])

                            if inter.area / box[1].area < MIN_INTERCECTION_RATIO:
                                continue

                            # get the smallest polygon (with sides parallel to the axes) that contains the intersection
                            new_box = inter.envelope

                            # get central point for the new bounding box
                            centre = new_box.centroid

                            # get coordinates of polygon vertices
                            x, y = new_box.exterior.coords.xy

                            if not (1 / MAX_HIGHT_WIDTH_RATIO < float(max(x) - min(x)) /
                                    (max(y) - min(y)) < MAX_HIGHT_WIDTH_RATIO):
                                continue

                            # get bounding box width and height normalized to slice size
                            new_width = (max(x) - min(x)) / SLIZE_SIZE
                            new_height = (max(y) - min(y)) / SLIZE_SIZE

                            # we have to normalize central x and invert y for yolo format
                            new_x = (centre.coords.xy[0][0] - x1) / SLIZE_SIZE
                            new_y = (y1 - centre.coords.xy[1][0]) / SLIZE_SIZE

                            counter += 1

                            slice_labels.append([box[0], new_x, new_y, new_width, new_height])
                            if not imsaved:
                                sliced = imr[i * SLIZE_SIZE:(i + 1) * SLIZE_SIZE, j * SLIZE_SIZE:(j + 1) * SLIZE_SIZE]
                                sliced_im = Image.fromarray(sliced)
                                filename = imname.split('/')[-1]
                                slice_path = f"{new_images_path}/{filename.replace('.jpg', f'_{i}_{j}.jpg')}"

                                slice_labels_path = f"{new_labels_path}/{filename.replace('.jpg', f'_{i}_{j}.txt')}"
                                print(slice_path)
                                sliced_im.save(slice_path)
                                imsaved = True
                                image_count += 1

                    # save txt with labels for the current tile
                    if len(slice_labels) > 0:
                        slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                        print(slice_df)
                        slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

                    # if there are no bounding boxes intersect current tile, save this tile to a separate folder
                    if not imsaved and image_count != 0 and float(
                            background_count) / image_count < BACKGROUND_IMAGES_RATIO:
                        sliced = imr[i * SLIZE_SIZE:(i + 1) * SLIZE_SIZE, j * SLIZE_SIZE:(j + 1) * SLIZE_SIZE]
                        sliced_im = Image.fromarray(sliced)
                        filename = imname.split('/')[-1]
                        slice_path = f"{new_images_path}/{filename.replace('.jpg', f'_{i}_{j}.jpg')}"
                        background_count += 1

                        sliced_im.save(slice_path)
                        print('Slice without boxes saved')
                        imsaved = True


if __name__ == '__main__':
    slice_dataset()
