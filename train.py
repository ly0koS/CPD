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


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    return image


def make_dataset(data_root):
    data_root = pathlib.Path(data_root)
    all_image_paths = list(data_root.glob("*/*"))
    all_image_paths = [str(p) for p in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    print("Loading %d pictures", image_count)
    label_names = sorted(
        item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index)
                          for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image,
                           num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds=tf.data.Dataset.zip((image_ds,label_ds))




data_root = os.path.join(PATH, "plate/train/city")
make_dataset(data_root)
