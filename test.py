import tensorflow as tf
from tensorflow import keras
from train import train,PlateData
import numpy as np
import os
import sys
from cv2 import cv2

Model_Path="/home/ly0kos/Car/model/1/"
AUTOTUNE = tf.data.experimental.AUTOTUNE
index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64}

try:
    model=keras.models.load_model(Model_Path)
except :
    print("load Model Error!\nTrying to train first!\n")
    train()




def gen_image_dataset(path):
    img_path=[]
    for root,dirs,files in os.walk(path):
        for name in files:
            img_path.append(name)
    img_set=[]
    test=1
    for i in img_path:
        i=os.path.join(path,i)
        image=cv2.imread(i)
        image=cv2.resize(image,(64,64))
        img_set.append(image)
    img_set=np.asarray(img_set)
    img_set=np.true_divide(img_set,255.0)

    return img_set

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = str()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys=item[0]
            return  listOfKeys

#PlateData(5000,273,76,1)
test_dataset=gen_image_dataset("/home/ly0kos/Car/temp/")
result=model.predict(test_dataset,verbose=1)
result=np.asarray(result)



for i in range(0,7):
    key=np.where(result[i][687]==np.amax(result[i][687]))
    key=np.asscalar(key[0])
    key=getKeysByValue(index,key)
    if i<6:
        print(key,end=' ')
    elif i==6:
        print(key)




