import os
import tensorflow as tf
from tensorflow import keras
import sys
import cv2
import numpy as np
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE


BATCH_SIZE = 500

dict={
    '京':0, '津':1, '沪':2, '渝':3, '冀':4, '豫':5, '云':6, 
    '辽':7, '黑':8, '湘':9, '皖':10, '鲁':11, '新':12, 
    '苏':13, '浙':14,'赣':15, '鄂':16, '桂':17, '甘':18, 
    '晋':19, '蒙':20, '陕':21,'吉':22, '闽':23, '贵':24, 
    '粤':25, '青':26, '藏':27, '川':28, '宁':29, '琼':30,
    "0":31, "1":32, "2":33, "3":34, "4":35, "5":36, "6":37, "7":38, "8":39, "9":40, 
    "A":43,"B":44, "C":45, "D":46, "E":47, "F":48, "G":49, "H":50, "I":51,"J":52, "K":53, 
    "L":54, "M":55,"N":56, "O":57, "P":58, "Q":59, "R":60, "S":61, "T":62, "U":63, "V":64, 
    "W":65, "X":66,"Y":67, "Z":68
}

def prepare_dataset(path):
    data_root=pathlib.Path(path)
    image_paths=list(data_root.glob('*'))
    image_paths = [str(path) for path in image_paths]
    image_count = len(image_paths)
    image_label=[]
    label_names=[]
    for item in data_root.glob("*.jpg"):
        name=item.name[0:7]
        temp=''
        for t in name:
            lab=str(dict[t])
            temp=temp+"%s"%lab
        temp=int(temp)
        image_label.append(temp)
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(image_label, tf.int64))
    image_ds=path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    steps_per_epoch=tf.math.ceil(len(image_paths)/BATCH_SIZE).numpy()


    return image_label_ds,image_count,steps_per_epoch

def preprocess_image(image):
    image=tf.image.decode_jpeg(image)
    image=tf.image.resize(image,[128,128])
    image/=255.0
    return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

def Forward():
       model=tf.keras.Sequential()
       model.add(tf.keras.layers.Conv2D(24,(3,3),activation="relu",padding='same',input_shape=(128,128,3),use_bias=True,kernel_initializer="he_normal"))
       model.add(tf.keras.layers.MaxPooling2D((2,2),strides=2))
       model.add(tf.keras.layers.Conv2D(48,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(tf.keras.layers.MaxPooling2D((2,2),strides=2))
       model.add(tf.keras.layers.Conv2D(96,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(tf.keras.layers.MaxPooling2D((2,2),strides=2))
       model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding='same',kernel_initializer="he_normal"))
       model.add(tf.keras.layers.Flatten())
       model.add(tf.keras.layers.Dense(128, activation='relu'))
       model.add(tf.keras.layers.Dense(10, activation='softmax'))

       return model


train_path="D:\\tf\images\\"
test_path="E:\TF\lpr_imgs\chepai\images"
(train_dataset,image_count,steps)=prepare_dataset(train_path)
test_dataset=prepare_dataset(test_path)

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = train_dataset
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(buffer_size=AUTOTUNE)

model=Forward()
model.compile(optimizer='rmsprop',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

model.fit(ds,epochs=10,steps_per_epoch=steps)

model.save('D:\\tf\\carplate',overwrite=True)
