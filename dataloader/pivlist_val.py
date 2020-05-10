import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    l0_train = []
    l1_train = []
    flow_train = []
    val = sorted(glob.glob(os.path.join(filepath, '*_flow.flo')))
    val = [img for img in val if int(img.split('_')[-2]) % 5 == 0]
    for flow_map in val:
        root_filename = flow_map[:-9]
        img1 = root_filename + '_img1.tif'
        img2 = root_filename + '_img2.tif'
        if not (os.path.isfile(os.path.join(filepath, img1)) and os.path.isfile(os.path.join(filepath, img2))):
            continue

        l0_train.append(img1)
        l1_train.append(img2)
        flow_train.append(flow_map)

    return l0_train, l1_train, flow_train
