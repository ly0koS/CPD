from cv2 import cv2
import os,sys
import tensorflow as tf
import numpy as np
from keras import layers as layers
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


def ReadData(path):
       L=[]
       num=0
       i=0
       if os.path.exists(path):
              for root,dirs,files in os.walk(path):
                     for name in files:
                            L.append(name)
                            num+=1
              pass
       train_data=np.zeros((num,480,480,1))
       while i<len(L):
              dir=path+str(L[i])
              image=tf.io.read_file(dir)
              image=tf.image.decode_jpeg(image)
              image=tf.cast(image,tf.float32)/255.0
              train_data[i]=image
              i+=1
       return(train_data)
       pass

def Forward(files):
       model=Sequential()
       model.add(Conv2D(24,(3,3),activation="relu",padding='same',data_format="channels_last",input_shape=(480,480,1)))
       model.add(MaxPooling2D((2,2),strides=2))
       model.add(Conv2D(48,(3,3),activation="relu",padding='same'))
       model.add(MaxPooling2D((2,2),strides=2))
       model.add(Conv2D(48,(3,3),activation="relu",padding='same'))
       model.add(layers.Flatten())
       model.add(layers.Dense(48, activation='relu'))
       model.add(layers.Dense(10, activation='softmax'))

       model.summary()

       model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
       return model

       pass

def main():
       train_data=ReadData("output/")
       Forward(train_data)
       pass

if __name__ == '__main__':
    main()
