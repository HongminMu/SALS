# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import glob
import math
import json
import os
import os.path as osp
import numpy as np
import PIL.Image
import PIL.ImageDraw
import cv2


#  --input_dir C:\Users\11385\Desktop\24-06-18

# --input_dir C:\Users\11385\Desktop\LLimages4_2  ICRA下面

# --input_dir C:\Users\11385\Desktop\24-08-11\output\refine  ICRA上面

# --input_dir C:\Users\11385\Desktop\24-05-23\2025-02-06\converted_jpeg


# Cityscapes color map (RGB)
CITYSCAPES_COLORS = [
    [128, 64, 128],  # 1: road
    [244, 35, 232],  # 2: sidewalk
    [70, 70, 70],    # 3: building
    [102, 102, 156], # 4: wall
    [190, 153, 153], # 5: fence
    [153, 153, 153], # 6: pole
    [250, 170, 30],  # 7: traffic_light
    [220, 220, 0],   # 8: traffic_sign
    [107, 142, 35],  # 9: vegetation
    [152, 251, 152], # 10: terrain
    [70, 130, 180],  # 11: sky
    [220, 20, 60],   # 12: person
    [255, 0, 0],     # 13: rider
    [0, 0, 142],     # 14: car
    [0, 0, 70],      # 15: truck
    [0, 60, 100],    # 16: bus
    [0, 80, 100],    # 17: train
    [0, 0, 230],     # 18: motorcycle
    [119, 11, 32]    # 19: bicycle
]

# Ensure all 19 class names are included
DEFAULT_CLASS_NAMES = [
    '_background_', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', help='input annotated directory')
    return parser.parse_args()


def main(args):
    output_dir = osp.join(args.input_dir, 'annotations')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print('Creating annotations directory:', output_dir)

    class_names = DEFAULT_CLASS_NAMES.copy()
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    print('class_names:', class_names)

    out_class_names_file = osp.join(args.input_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    color_map = get_cityscapes_color_map()

    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_png_file = osp.join(output_dir, base + '.png')

            data = json.load(f)

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(cv2.imread(img_file))

            lbl = shape2label(
                img_size=img.shape,
                shapes=data['shapes'],
                class_name_mapping=class_name_to_id, )

            if osp.splitext(out_png_file)[1] != '.png':
                out_png_file += '.png'
            # Assume label ranges [0, 255] for uint8,
            if lbl.min() >= 0 and lbl.max() <= 255:
                lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)
                lbl_pil.save(out_png_file)
            else:
                raise ValueError(
                    '[%s] Cannot save the pixel-wise class label as PNG. '
                    'Please consider using the .npy format.' % out_png_file)


def get_cityscapes_color_map():
    color_map = np.zeros((256, 3), dtype=np.uint8)
    for i, color in enumerate(CITYSCAPES_COLORS):
        color_map[i + 1] = color  # class ids start from 1
    return color_map.flatten()


def shape2mask(img_size, points):
    label_mask = PIL.Image.fromarray(np.zeros(img_size[:2], dtype=np.uint8))
    image_draw = PIL.ImageDraw.Draw(label_mask)
    points_list = [tuple(point) for point in points]
    assert len(points_list) > 2, 'Polygon must have points more than 2'
    image_draw.polygon(xy=points_list, outline=1, fill=1)
    return np.array(label_mask, dtype=bool)


def shape2label(img_size, shapes, class_name_mapping):
    label = np.zeros(img_size[:2], dtype=np.int32)
    for shape in shapes:
        points = shape['points']
        if len(points) <= 2:
            print(f"Skipping invalid shape with points < 3: {points}")
            continue
        class_name = shape['label']
        shape_type = shape.get('shape_type', None)
        class_id = class_name_mapping[class_name]
        label_mask = shape2mask(img_size[:2], points)
        label[label_mask] = class_id
    return label


if __name__ == '__main__':
    args = parse_args()
    main(args)
