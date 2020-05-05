#coding:utf-8
import tensorflow as tf
from tensorflow import keras
from train import train
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from processPicture import gen_dataset

Model_Path="/home/ly0kos/tensorflow/CPD/model/"
AUTOTUNE = tf.data.experimental.AUTOTUNE

write_path="/home/ly0kos/WD/tensorflow/CCPD2019/ccpd_test/"

index={
  "A" : 0,"B" : 1,"C" : 2,"D" : 3,"E" : 4,"F" : 5,"G" : 6,"H" : 7,"J" : 8,"K" : 9,"L" : 10,"M" : 11,
  "N" : 12,"P" : 13,"Q" : 14,"R" : 15,"S" : 16,"T" : 17,"U" : 18,"V" : 19,"W" : 20,"X":  21,
  "Y" : 22,"Z" : 23,"0" : 24,"1" : 25,"2" : 26,"3" : 27,"4" : 28,"5" : 29,"6" : 30,"7" : 31,
  "8" : 32,"9" : 33,
  "皖": 34,"沪": 35,"津": 36,"渝": 37,"冀": 38,"晋": 39,"蒙": 40,"辽": 41,"吉": 42,"黑": 43,"苏": 44,"浙": 45,
  "京": 46,"闽": 47,"赣": 48,"鲁": 49,"豫": 50,"鄂": 51,"湘": 52,"粤": 53,"桂": 54,"琼": 55,
  "川": 56,"贵": 57,"云": 58,"西": 59,"陕": 60,"甘": 61,"青": 62,"宁": 63,"新": 64
}

try:
    model=keras.models.load_model(Model_Path)
except :
    print("load Model Error!\nTrying to train first!\n")
    train()
    model=keras.models.load_model(Model_Path)




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

    return img_set,img_path

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = str()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys=item[0]
            return  listOfKeys

#PlateData(5000,273,76,1)
count=1000
path_tmp="/home/ly0kos/WD/tensorflow/CCPD2019/ccpd_challenge/"
path=[]
path.append(path_tmp)
test_dataset,label_dataset=gen_dataset(path,count,2)
result=model.predict(test_dataset,verbose=1)
result=np.asarray(result)


for i in range(0,10):
    ran=np.random.randint(0,count)
    predict_key=str()
    for j in range(0,7):
            key=np.where(result[j][ran]==np.amax(result[j][ran]))                                                   #result[charlocate][#plate]
            key=np.asscalar(key[0])
            key=getKeysByValue(index,key)
            predict_key=predict_key+key
    char=str()  
    for i in range(0,7):
        char+=getKeysByValue(index,label_dataset[ran][i])                                                  #decode filename
    plt.imshow(test_dataset[ran])
    plt.title(u'真实车牌号:'+char)
    plt.xlabel(predict_key)
    plt.show()