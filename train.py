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
from processPicture import gen_dataset
import gc
AUTOTUNE = tf.data.experimental.AUTOTUNE
PATH = "/home/ly0kos/Car/"
SAVE_PATH="/home/ly0kos/tensorflow/CPD/model/"
BATCH_SIZE = 20

def PlateData(path):
    data=[]
    dataset=gen_dataset(path,1)
    gc.collect()
    dataset=dataset.shuffle(buffer_size=BATCH_SIZE)
    #dataset=dataset.repeat()                                                                                                   #big dataset,disable to prevent OOM
    dataset=dataset.batch(BATCH_SIZE)
    dataset=dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset    
        

def Forward():
    input=keras.Input(shape=(128,128,3),name='title')
    x=layers.Conv2D(16,3,activation="relu",padding='same',kernel_initializer="he_normal")(input)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Dropout(0.25)(x)
    x=layers.Conv2D(32,3,activation="relu",padding='same',kernel_initializer="he_normal")(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Conv2D(64,3,activation="relu",padding='same',kernel_initializer="he_normal")(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Conv2D(128,3,activation="relu",padding='same',kernel_initializer="he_normal")(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Dropout(0.20)(x)
    x=layers.Flatten()(x)
    output_Zh=layers.Dense(65,activation="softmax",name="output_Zh")(x)
    output_1=layers.Dense(65,activation="softmax",name="output_Ch1")(x)
    output_2=layers.Dense(65,activation="softmax",name="output_Ch2")(x)
    output_3=layers.Dense(65,activation="softmax",name="output_Ch3")(x)
    output_4=layers.Dense(65,activation="softmax",name="output_Ch4")(x)
    output_5=layers.Dense(65,activation="softmax",name="output_Ch5")(x)
    output_6=layers.Dense(65,activation="softmax",name="output_Ch6")(x)
    model=keras.Model(inputs=input,outputs=[output_Zh,output_1,output_2,output_3,output_4,output_5,output_6])
    
    return model


def train():
    path="/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_rotate"
    count=len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    dataset=PlateData(path)
    
    model=Forward()
    model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False,name="loss"),
                metrics=["accuracy"]) 
    steps=tf.math.ceil(count/BATCH_SIZE).numpy()

    keras.utils.plot_model(model, 'model.png', show_shapes=True)

    model.summary()

    model.fit(dataset,steps_per_epoch=steps,epochs=20)


    model.save(SAVE_PATH)