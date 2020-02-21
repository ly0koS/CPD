import os
import tensorflow as tf
import sys
import cv2
import random
import numpy as np

PATH="D:\\tf\images\\"
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

def load_img(width,height):
    img=[]
    label=[]
    image_path=[]
    num=os.listdir(PATH)
    for i in num:
        filename=PATH+"%s"%i
        image = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
        image=cv2.resize(image,(width,height))
        img.append(image)

        temp=""
        for t in i[0:7]:
            lab=str(dict[t])
            temp=temp+"%s"%lab
        label.append(temp)
        
    data=np.array(img)
    label=np.array(label)


    return data,label


(data,label)=load_img(255,128)