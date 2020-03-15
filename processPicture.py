from cv2 import cv2
import os
import numpy as np

Zh={
  "皖": 0,"沪": 1,"津": 2,"渝": 3,"冀": 4,"晋": 5,"蒙": 6,"辽": 7,"吉": 8,"黑": 9,"苏": 10,"浙": 11,
  "京": 12,"闽": 13,"赣": 14,"鲁": 15,"豫": 16,"鄂": 17,"湘": 18,"粤": 19,"桂": 20,"琼": 21,
  "川": 22,"贵": 23,"云": 24,"西": 25,"陕": 26,"甘": 27,"青": 28,"宁": 29,"新": 30
}

Char={
  "A" : 0,"B" : 1,"C" : 2,"D" : 3,"E" : 4,"F" : 5,"G" : 6,"H" : 7,"J" : 8,"K" : 9,"L" : 10,"M" : 11,
  "N" : 12,"P" : 13,"Q" : 14,"R" : 15,"S" : 16,"T" : 17,"U" : 18,"V" : 19,"W" : 20,"X":  21,
  "Y" : 22,"Z" : 23,"0" : 24,"1" : 25,"2" : 26,"3" : 27,"4" : 28,"5" : 29,"6" : 30,"7" : 31,
  "8" : 32,"9" : 33
}

def find_char(string,key,start,time):
    temp=start+1
    for t in range(0,time):
        temp=string.find(key,temp+1)
    return temp

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = str()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys=item[0]
            return  listOfKeys

def load_img(location,folder):
    image=cv2.imread(location)
    start=len(folder)+1
    x_start=find_char(location,"-",start,2)
    x_end=find_char(location,"&",start,1)
    x=location[x_start+1:x_end]
    x=int(x)
    y_end=find_char(location,"_",start,2)
    y=location[x_end+1:y_end]
    y=int(y)
    x1_end=find_char(location,"&",start,2)
    x1=location[y_end+1:x1_end]
    x1=int(x1)
    y1_end=find_char(location,"-",start,3)
    y1=location[x1_end+1:y1_end]
    y1=int(y1)
    image=image[y:y1,x:x1]
    return image

def load_label(location):
    key_list=[]
    start=find_char(location,'-',-50,1)
    key=location[start+1:start+2]
    key=int(key)+34
    key_list.append(key)
    for i in range(0,5):
        start=find_char(location,'_',start,1)
        end=find_char(location,'_',start,1)
        key=location[start+1:end]
        key=int(key)
        key_list.append(key)
    last=find_char(location,'-',start,1)
    key=location[end+1:last]
    key=int(key)
    key_list.append(key)
    return key_list

def gen_dataset(path):
    img_path=[]
    img_data=[]
    for root,dir,filenames in os.walk(path):
        for name in filenames:
            name=os.path.join(path,name)
            img_path.append(name)
    label_data=np.empty((len(img_path),7))
    num=0
    for loc in img_path:
        image=load_img(loc,path)
        image=cv2.resize(image,(128,128))
        img_data.append(image)
        label=load_label(loc)
        label=np.asarray(label)
        label_data[num]=label
        cv2.imwrite("/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_train/"+str(label)+".jpg",image)
        num+=1
    img_data=np.asarray(img_data)
    return  img_data,label_data


