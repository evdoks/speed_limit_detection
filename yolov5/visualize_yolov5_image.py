import yaml
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageColor, ImageFont


def visualize_gt(image_key, dataset_dict, color='green', alpha=125, font=None):
    try:
        font = ImageFont.truetype('arial.ttf', 15)
    except Exception:
        print('Falling back to default font...')
        font = ImageFont.load_default()

    with Image.open(os.path.join(os.path.dirname(DATASET_YAML), 'images', set_key,
                                 '{:s}.jpg'.format(image_key))) as img:
        img = img.convert('RGBA')
        img_draw = ImageDraw.Draw(img)

        rects = Image.new('RGBA', img.size)
        rects_draw = ImageDraw.Draw(rects)

        width, height = img.size
        labname = os.path.join(os.path.dirname(DATASET_YAML), 'labels', set_key, '{:s}.txt'.format(image_key))
        labels = pd.read_csv(labname, sep=' ', names=['label', 'x1', 'y1', 'w', 'h'])

        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height

        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w'] / 2
            y1 = row[1]['y1'] - row[1]['h'] / 2
            x2 = row[1]['x1'] + row[1]['w'] / 2
            y2 = row[1]['y1'] + row[1]['h'] / 2

            color_tuple = ImageColor.getrgb(color)
            if len(color_tuple) == 3:
                color_tuple = color_tuple + (alpha, )
            else:
                color_tuple[-1] = alpha

            rects_draw.rectangle((x1 + 1, y1 + 1, x2 - 1, y2 - 1), fill=color_tuple)
            img_draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='black', width=1)

            class_name = dataset_dict['names'][labels['label'].values[0]]
            img_draw.text((x1 + 5, y1 + 5), class_name, font=font)

        img = Image.alpha_composite(img, rects)

        return img


if __name__ == '__main__':
    DATASET_YAML = '../../datasets/mtsd_speed_limits/mtsd.yaml'
    set_key = 'val'
    image_key = 'xO0x7fQBfTpu-p6BHOKSXQ'

    with open(DATASET_YAML) as f:
        dataset_dict = yaml.safe_load(f)

    # visualize traffic sign boxes on the image
    vis_img = visualize_gt(image_key, dataset_dict)
    vis_img.show()
