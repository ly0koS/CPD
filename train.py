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
SAVE_PATH="/home/ly0kos/Car/model/"
BATCH_SIZE = 20

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [64, 64])
    image /= 255.0  # normalize to [0,1] range
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
    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
    # 被充分打乱。
    ds = image_label_ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
    return label_to_index,ds,steps_per_epoch

def Forward():
       model=tf.keras.Sequential()
       model.add(tf.keras.layers.Conv2D(8,(3,3),activation="relu",padding='same',input_shape=(64,64,3),kernel_initializer="he_normal"))
       model.add(tf.keras.layers.MaxPooling2D((2,2),strides=2))
       model.add(tf.keras.layers.Dropout(0.2))
       model.add(tf.keras.layers.Conv2D(16,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(tf.keras.layers.MaxPooling2D((2,2),strides=2))
       model.add(tf.keras.layers.Conv2D(32,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(tf.keras.layers.MaxPooling2D((2,2),strides=2))
       model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(tf.keras.layers.MaxPooling2D((2,2),strides=2))
       model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(tf.keras.layers.Dropout(0.2))
       model.add(tf.keras.layers.Flatten())
       model.add(tf.keras.layers.Dense(64, activation='relu'))
       model.add(tf.keras.layers.Dense(32, activation='softmax'))

       return model



data_root = os.path.join(PATH, "plate/train/city")
label,ds,steps_per_epoch = make_dataset(data_root)



Han_model=Forward()
Han_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

save_model=os.path.join(SAVE_PATH,"Han/")

Han_model.summary()

Han_model.fit(ds,epochs=25,steps_per_epoch=steps_per_epoch)

Han_model.save(save_model,overwrite=True) 



""" img=os.path.join("/home/ly0kos/Car/plate/train/city/鄂","鄂A529HI.jpg")
img=load_and_preprocess_image(img)
img = np.expand_dims(img, axis=0)
predict=Han_model.predict_classes(img)

for (k,v) in label.items():
    if v==predict:
        print(k) """