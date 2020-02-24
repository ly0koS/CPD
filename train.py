from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import pathlib
import tensorflow as tf
import numpy as np
from cv2 import cv2
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE
PATH = "/home/ly0kos/Car/"
label = {
    '京': 0, '津': 1, '沪': 2, '渝': 3, '冀': 4, '豫': 5, '云': 6,
    '辽': 7, '黑': 8, '湘': 9, '皖': 10, '鲁': 11, '新': 12,
    '苏': 13, '浙': 14, '赣': 15, '鄂': 16, '桂': 17, '甘': 18,
    '晋': 19, '蒙': 20, '陕': 21, '吉': 22, '闽': 23, '贵': 24,
    '粤': 25, '青': 26, '藏': 27, '川': 28, '宁': 29, '琼': 30,
    "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
    "A": 43, "B": 44, "C": 45, "D": 46, "E": 47, "F": 48, "G": 49, "H": 50, "I": 51, "J": 52, "K": 53,
    "L": 54, "M": 55, "N": 56, "O": 57, "P": 58, "Q": 59, "R": 60, "S": 61, "T": 62, "U": 63, "V": 64,
    "W": 65, "X": 66, "Y": 67, "Z": 68
}


def make_dataset(data_root):
    data_root = pathlib.Path(data_root)
    all_image_paths = list(data_root.glob("*/*"))
    all_image_paths = [str(p) for p in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    label_names = sorted(
        item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
    
    return all_image_labels


data_root=os.path.join(PATH, "plate/train/city")
make_dataset(data_root)
