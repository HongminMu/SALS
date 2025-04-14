import os
import math
import json
import argparse
import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.core import infer
from eiseg.util.polygon import get_polygon
from paddleseg.utils import logger, progbar, visualize
from paddleseg.cvlibs import manager, Config, SegBuilder
from paddleseg.utils import get_sys_env, logger, get_image_list
from paddleseg.core import predict
from paddleseg.transforms import Compose
# --config D:\PaddleSeg-release-2.8\configs/mscale_ocrnet/mscale_ocrnet_hrnetv2_psa_cityscapes_1024x2048_150k.yml --model_path D:\PaddleSeg-release-2.8\models/Mscale_ocrnet/model.pdparams --image_path C:\Users\11385\Desktop\LLimages2 --save_dir C:\Users\11385\Desktop\LLimages4_0
# --config D:\PaddleSeg-release-2.8\configs/mscale_ocrnet/mscale_ocrnet_hrnetv2_psa_cityscapes_1024x2048_150k.yml --model_path D:\PaddleSeg-release-2.8\models/Mscale_ocrnet/model.pdparams --image_path C:\Users\11385\Desktop\24-06-18 --save_dir C:\Users\11385\Desktop\24-06-18\output

# --config D:\PaddleSeg-release-2.8\configs/mscale_ocrnet/mscale_ocrnet_hrnetv2_psa_cityscapes_1024x2048_150k.yml --model_path D:\PaddleSeg-release-2.8\models/Mscale_ocrnet/model.pdparams --image_path C:\Users\11385\Desktop\24-06-19 --save_dir C:\Users\11385\Desktop\24-06-19\output
# --config D:\PaddleSeg-release-2.8\configs/mscale_ocrnet/mscale_ocrnet_hrnetv2_psa_cityscapes_1024x2048_150k.yml --model_path D:\PaddleSeg-release-2.8\models/Mscale_ocrnet/model.pdparams --image_path C:\Users\11385\Desktop\24-05-23\ICRA2025\Dark_Zurich_Samples  --save_dir C:\Users\11385\Desktop\24-08-11\output

def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of trained weights for prediction.',
        type=str)
    parser.add_argument(
        '--image_path',
        help='The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the predicted results.',
        type=str,
        default='./output/result')
    parser.add_argument(
        '--device',
        help='Set the device place for predicting model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)

    # Data augment params
    parser.add_argument(
        '--aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        nargs='+',
        help='Scales for augment, e.g., `--scales 0.75 1.0 1.25`.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # Sliding window evaluation params
    parser.add_argument(
        '--is_slide',
        help='Whether to predict images in sliding window method',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.'
        'For example, `--crop_size 512 512`',
        type=int)
    parser.add_argument(
        '--stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.'
        'For example, `--stride 512 512`',
        type=int)

    # Custom color map
    parser.add_argument(
        '--custom_color',
        nargs='+',
        help='Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int)

    return parser.parse_args()


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if 'aug_eval' in test_config:
        test_config.pop('aug_eval')
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    if args.custom_color:
        test_config['custom_color'] = args.custom_color
    return test_config


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def preprocess(im_path, transforms):
    data = {}
    data['img'] = im_path
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = paddle.to_tensor(data['img'])
    return data


# convert various types of data into JSON format
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            return super(NpEncoder, self).default(obj)


def get_polygons_for_all_classes(pred, img_size):
    all_polygons = {}  # 初始化存储所有类别多边形的字典

    for class_id in range(19):  # 假设有19个类别
        # 创建二值图像，前景为255，背景为0
        class_mask = np.where(pred == class_id, 255, 0).astype(np.uint8)
        class_polygons = get_polygon(class_mask, img_size=img_size, building=False)  # 获取当前类别的多边形轮廓
        if class_polygons is not None:  # 检查class_polygons是否为None
            if class_id not in all_polygons:
                all_polygons[class_id] = []  # 如果字典中还没有这个类别，则初始化一个空列表
            all_polygons[class_id].extend(class_polygons)  # 添加当前类别的多边形

    return all_polygons

# 类别ID到名称的映射
class_id_to_name = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle"
}

# 类别ID到颜色的映射
class_id_to_color = {
    0: [128, 64, 128],
    1: [244, 35, 232],
    2: [70, 70, 70],
    3: [102, 102, 156],
    4: [190, 153, 153],
    5: [153, 153, 153],
    6: [250, 170, 30],
    7: [220, 220, 0],
    8: [107, 142, 35],
    9: [152, 251, 152],
    10: [70, 130, 180],
    11: [220, 20, 60],
    12: [255, 0, 0],
    13: [0, 0, 142],
    14: [0, 0, 70],
    15: [0, 60, 100],
    16: [0, 80, 100],
    17: [0, 0, 230],
    18: [119, 11, 32]
}

def get_custom_color_map(class_id_to_color):
    color_map = np.zeros((256, 3), dtype=np.uint8)
    for class_id, color in class_id_to_color.items():
        color_map[class_id] = color
    return color_map.flatten().tolist()


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

    polygons = []
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = get_custom_color_map(class_id_to_color)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            data = preprocess(im_path, transforms)

            if aug_pred:
                pred, _ = infer.aug_inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred, _ = infer.inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # 获取所有类别的多边形
            all_class_polygons = get_polygons_for_all_classes(pred, img_size=pred.shape)

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            added_image = utils.visualize.visualize(
                im_path, pred, color_map, weight=0.6)
            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)
            cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            pred_saved_path = os.path.join(
                pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            progbar_pred.update(i + 1)

            # define the information required for a single image
            json_data = {
                "version": "5.4.1",
                "flags": {},
                "shapes": [],
                "imagePath": im_file,
                "imageData": None  # Here we keep it None, you can optionally add image data
            }

            for class_id, class_polygons in all_class_polygons.items():
                class_name = class_id_to_name[class_id]
                for polygon in class_polygons:
                    shape = {
                        "label": class_name,
                        "points": polygon,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None
                    }
                    json_data["shapes"].append(shape)
            # save individual JSON file for each image
            json_saved_name = os.path.join(save_dir, os.path.splitext(im_file)[0] + ".json")
            with open(json_saved_name, "w", encoding="utf-8") as f:
                json.dump(json_data, f, cls=NpEncoder)

    logger.info("Predicted images are saved in {} and {} .".format(
        added_saved_dir, pred_saved_dir))

def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config)
    builder = SegBuilder(cfg)
    test_config = merge_test_config(cfg, args)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_device(args.device)

    model = builder.model
    transforms = Compose(builder.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('The number of images: {}'.format(len(image_list)))

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        **test_config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
