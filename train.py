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
    label=[]
    data,label=genplate.genBatch(count,"/home/ly0kos/Car/temp",(height,width))
    data=np.asarray(data)
    label=np.asarray(label)
    print(label.shape)
    dataset=tf.data.Dataset.from_tensor_slices((data,label))
    dataset=dataset.shuffle(buffer_size=1000)
    dataset=dataset.batch(BATCH_SIZE)
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
    x=layers.Dropout(0.25)(x)
    output_Zh=layers.Dense(32)(x)
    output_1=layers.Dense(34)(x)
    output_2=layers.Dense(34)(x)
    output_3=layers.Dense(34)(x)
    output_4=layers.Dense(34)(x)
    output_5=layers.Dense(34)(x)
    output_6=layers.Dense(34)(x)
    model=keras.Model(inputs=input,outputs=[output_Zh,output_1,output_2,output_3,output_4,output_5,output_6])
    
    return model

dataset=PlateData(10000,64,64)

model=Forward()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

#model.fit(dataset,epochs=5)

#save_model=os.path.join(SAVE_PATH,"Han/")





