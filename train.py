from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from cv2 import cv2
import random
from Genplate import *

AUTOTUNE = tf.data.experimental.AUTOTUNE
PATH = "/home/ly0kos/Car/"
SAVE_PATH="/home/ly0kos/Car/model/"
BATCH_SIZE = 20

def PlateData(count, height, width):
    genplate=GenPlate("/home/ly0kos/Car/font/platech.ttf",'/home/ly0kos/Car/font/platechar.ttf')
    data=[]
    labelZh=np.empty((count,1,1))
    labelCh1=np.empty((count,1,1))
    labelCh2=np.empty((count,1,1))
    labelCh3=np.empty((count,1,1))
    labelCh4=np.empty((count,1,1))
    labelCh5=np.empty((count,1,1))
    labelCh6=np.empty((count,1,1))
    data,label=genplate.genBatch(count,"/home/ly0kos/Car/temp",(height,width))
    data=np.asarray(data)
    data=data/255.0
    for i in range(0,count):
        labelZh[i]=label[i][0]
        labelCh1[i]=label[i][1]
        labelCh2[i]=label[i][2]
        labelCh3[i]=label[i][3]
        labelCh4[i]=label[i][4]
        labelCh5[i]=label[i][5]
        labelCh6[i]=label[i][6]
    dataset=tf.data.Dataset.from_tensor_slices((
        data,
        {
            'dense_1': labelZh,
            'dense_2': labelCh1,
            'dense_3': labelCh2,
            'dense_4': labelCh3,
            'dense_5': labelCh4,
            'dense_6': labelCh5,
            'dense_7': labelCh6
        }
        ))
    dataset=dataset.shuffle(buffer_size=1000)
    dataset=dataset.repeat()
    dataset=dataset.batch(BATCH_SIZE)
    dataset=dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset    
        

def Forward():
    input=keras.Input(shape=(64,64,3),name='title')
    x=layers.Conv2D(8,3,activation="relu",padding='same',kernel_initializer="he_normal")(input)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Dropout(0.25)(x)
    x=layers.Conv2D(16,3,activation="relu",padding='same',kernel_initializer="he_normal")(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Conv2D(32,3,activation="relu",padding='same',kernel_initializer="he_normal")(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Conv2D(64,3,activation="relu",padding='same',kernel_initializer="he_normal")(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Conv2D(128,3,activation="relu",padding='same',kernel_initializer="he_normal")(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Conv2D(256,3,activation="relu",padding='same',kernel_initializer="he_normal")(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Dropout(0.20)(x)
    x=layers.Flatten()(x)
    x=layers.Dense(120,activation="relu")(x)
    output_Zh=layers.Dense(65,activation="softmax")(x)
    output_1=layers.Dense(65,activation="softmax")(x)
    output_2=layers.Dense(65,activation="softmax")(x)
    output_3=layers.Dense(65,activation="softmax")(x)
    output_4=layers.Dense(65,activation="softmax")(x)
    output_5=layers.Dense(65,activation="softmax")(x)
    output_6=layers.Dense(65,activation="softmax")(x)
    model=keras.Model(inputs=input,outputs=[output_Zh,output_1,output_2,output_3,output_4,output_5,output_6])
    
    return model


dataset=PlateData(10000,64,64)

model=Forward()
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False,name="loss"),
              metrics=["accuracy"]) 
steps=tf.math.ceil(10000/BATCH_SIZE).numpy()
model.fit(dataset,steps_per_epoch=steps,epochs=20)

save_model=os.path.join(SAVE_PATH,"1/")

tf.saved_model.save(model,save_model)


