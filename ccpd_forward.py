from cv2 import cv2
import os,sys
import tensorflow as tf
import numpy as np
from keras import layers as layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

INPUT_NODE = 835200

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def ReadData(path):
       labels=[]
       num=0
       i=0
       if os.path.exists(path):
              for root,dirs,files in os.walk(path):
                     for name in files:
                            labels.append(name)
                            num+=1
       labels=np.array(labels)
       data=np.zeros((num,72,272,3))
       while i<len(labels):
              dir=path+str(labels[i])
              image=tf.io.read_file(dir)
              image=tf.image.decode_jpeg(image)
              image=tf.cast(image,tf.float32)/255.0
              data[i]=image
              i+=1
       shape=labels.shape[0]
       labels=np.reshape(labels,(shape,1))
       return(data,labels)

def Forward(files):
       model=Sequential()
       model.add(Conv2D(24,(3,3),activation="relu",padding='same',input_shape=(72,272,3),use_bias=True,kernel_initializer="he_normal"))
       model.add(MaxPooling2D((2,2),strides=2))
       model.add(Conv2D(48,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(MaxPooling2D((2,2),strides=2))
       model.add(Conv2D(96,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(MaxPooling2D((2,2),strides=2))
       model.add(Conv2D(128,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(layers.Flatten())
       model.add(layers.Dense(48, activation='relu'))
       model.add(layers.Dense(10, activation='softmax'))

       model.summary()

       model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

       return model

def main():
       test_data=ReadData("plate/")
       train_data,train_lables=ReadData("plate_test/")
       model=Forward(test_data)
       model.fit(train_data,train_lables,epochs=10)
       

if __name__ == '__main__':
    main()
