import cv2
import os,sys
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


INPUT_NODE = 835200

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def DataRead(path,L):
       if os.path.exists(path):
              for root,dirs,files in os.walk(path):
                     for name in files:
                            L.append(name)
              pass
       pass

def Forward(files):
       hide1=Sequential([
              Conv2D(filters=48,kernel_size=5,strides=1,padding="same",activation='relu',input_shape=(480,480,1) ),
              MaxPooling2D(pool_size=(2,2),strides=1,padding="same"),
              Dropout(rate=0.2)
              ] )
       
       pass

def main():
       files=[]
       DataRead("output/",files)
       Forward(files)
       pass

if __name__ == '__main__':
    main()
