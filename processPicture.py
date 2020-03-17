from cv2 import cv2
import os
import numpy as np
import gc
import tensorflow as tf
import sys
import random
import pathlib
AUTOTUNE = tf.data.experimental.AUTOTUNE

Zh = {
    "皖": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11,
    "京": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21,
    "川": 22, "贵": 23, "云": 24, "西": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30
}

Char = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "J": 8, "K": 9, "L": 10, "M": 11,
    "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "U": 18, "V": 19, "W": 20, "X":  21,
    "Y": 22, "Z": 23, "0": 24, "1": 25, "2": 26, "3": 27, "4": 28, "5": 29, "6": 30, "7": 31,
    "8": 32, "9": 33
}


def find_char(string, key, start, time):
    temp = start+1
    for t in range(0, time):
        temp = string.find(key, temp+1)
    return temp


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = str()
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if item[1] == valueToFind:
            listOfKeys = item[0]
            return listOfKeys


def find_crop_box(folder, location):
    start = len(folder)+1
    x_start = find_char(location, "-", start, 2)
    x_end = find_char(location, "&", start, 1)
    x = location[x_start+1:x_end]
    x = int(x)
    y_end = find_char(location, "_", start, 2)
    y = location[x_end+1:y_end]
    y = int(y)
    x1_end = find_char(location, "&", start, 2)
    x1 = location[y_end+1:x1_end]
    x1 = int(x1)
    y1_end = find_char(location, "-", start, 3)
    y1 = location[x1_end+1:y1_end]
    y1 = int(y1)
    return x, x1, y, y1


def crop_plate(location, folder):
    image = cv2.imread(location)
    x, x1, y, y1 = find_crop_box(folder, location)
    image = image[y:y1, x:x1]
    return image


def find_label(location):
    key_list = []
    start = find_char(location, '-', -50, 1)
    key = location[start+1:start+2]
    key = int(key)+34
    key_list.append(key)
    for i in range(0, 5):
        start = find_char(location, '_', start, 1)
        end = find_char(location, '_', start, 1)
        key = location[start+1:end]
        key = int(key)
        key_list.append(key)
    last = find_char(location, '-', start, 1)
    key = location[end+1:last]
    key = int(key)
    key_list.append(key)
    return key_list

def load_label(filename):
    key_list = []
    key_list.append(Zh[filename[0]]+34)
    for i in range(0,6):
        key_list.append(Char[filename[i+1]])
    return key_list

def preprocess(img_path, path, write_path,write_flag):
    num = 0
    for loc in img_path:
        label = find_label(loc)  # label of number
        label = np.asarray(label)
        image = crop_plate(loc, path)
        char = getKeysByValue(Zh, label[0]-34)  # decode filename
        for i in range(1, 7):
            char += getKeysByValue(Char, label[i])
        if not os.path.exists(write_path):
            os.mkdir(write_path)
        filename=os.path.join(write_path,char+".jpg")
        cv2.imwrite(filename, image)
        num += 1


def tf_preprocess_image(image):
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [128, 128])
    image = image/255.0
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return tf_preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

def gen_dataset(path, write_flag):
    img_path = []
    img_data = []
    for root, dir, filenames in os.walk(path):
        for name in filenames:
            name = os.path.join(path, name)
            img_path.append(name)
    label_data = np.empty((len(img_path), 7))
    num = 0
    if write_flag == 1:
        write_path = "/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_train/"
    elif write_flag == 2:
        write_path = "/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_test/"
    preprocess(img_path, path, write_path,write_flag)
    all_image_paths=[]
    file_name=[]
    for root, dir, filenames in os.walk(write_path):
        for name in filenames:
            file_name.append(name)
            name = os.path.join(write_path, name)
            all_image_paths.append(name)
    #random.shuffle(all_image_paths)
    for path in filenames:
        label = load_label(path)  # label of number
        label = np.asarray(label)
        label_data[num] = label
    count=len(all_image_paths)
    labelZh=np.empty((count,1,1))
    labelCh1=np.empty((count,1,1))
    labelCh2=np.empty((count,1,1))
    labelCh3=np.empty((count,1,1))
    labelCh4=np.empty((count,1,1))
    labelCh5=np.empty((count,1,1))
    labelCh6=np.empty((count,1,1))
    for i in range(0,count):
        labelZh[i]=label_data[i][0]
        labelCh1[i]=label_data[i][1]
        labelCh2[i]=label_data[i][2]
        labelCh3[i]=label_data[i][3]
        labelCh4[i]=label_data[i][4]
        labelCh5[i]=label_data[i][5]
        labelCh6[i]=label_data[i][6]
    dataset=tf.data.Dataset.from_tensor_slices((
        all_image_paths,
        {
            'output_Zh': labelZh,
            'output_Ch1': labelCh1,
            'output_Ch2': labelCh2,
            'output_Ch3': labelCh3,
            'output_Ch4': labelCh4,
            'output_Ch5': labelCh5,
            'output_Ch6': labelCh6
        }
        ))
    image_label_ds = dataset.map(load_and_preprocess_from_path_label)
    return image_label_ds
