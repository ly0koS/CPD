from cv2 import cv2
import os
import numpy as np

Zh={
  "皖": 0,"沪": 1,"津": 2,"渝": 3,"冀": 4,"晋": 5,"蒙": 6,"辽": 7,"吉": 8,"黑": 9,"苏": 10,"浙": 11,
  "京": 12,"闽": 13,"赣": 14,"鲁": 15,"豫": 16,"鄂": 17,"湘": 18,"粤": 19,"桂": 20,"琼": 21,
  "川": 22,"贵": 23,"云": 24,"西": 25,"陕": 26,"甘": 27,"青": 28,"宁": 29,"新": 30
}

Char={
  "a" : 0,"b" : 1,"c" : 2,"d" : 3,"e" : 4,"f" : 5,"g" : 6,"h" : 7,"j" : 8,"k" : 9,"l" : 10,"m" : 11,
  "n" : 12,"p" : 13,"q" : 14,"r" : 15,"s" : 16,"t" : 17,"u" : 18,"v" : 19,"w" : 20,"x":  21,
  "y" : 22,"z" : 23,"0" : 24,"1" : 25,"2" : 26,"3" : 27,"4" : 28,"5" : 29,"6" : 30,"7" : 31,
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
    wherestart=location.find("-",54)
    whereend=location.find("-",69)
    print(location[wherestart:whereend])

def gen_dataset(path):
    img_path=[]
    img_data=[]
    label_data=[]
    for root,dir,filenames in os.walk(path):
        for name in filenames:
            name=os.path.join(path,name)
            img_path.append(name)
    file_count=len(img_path)
    num=0
    for loc in img_path:
        image=load_img(loc,path)
        image=cv2.resize(image,(128,128))
        img_data.append(image)
        num+=1
    img_data=np.asarray(img_data)
    return  img_data,label_data


gen_dataset("/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_challenge")