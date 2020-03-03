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

class PlateData(tf.data.Dataset):
    def __init__(self,count, num_label, height, width):
        self.genplate = GenPlate("/home/ly0kos/Car/font/platech.ttf",'/home/ly0kos/Car/font/font/platechar.ttf')
        self.count = count
        self.height = height
        self.width = width
        data,label=self.genplate.genBatch(count,"/home/ly0kos/Car/temp",(height,width))
        
        
                

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




Han_model=Forward()
Han_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

"""save_model=os.path.join(SAVE_PATH,"Han/")"""

Han_model.summary()


