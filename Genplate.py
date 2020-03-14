import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from cv2 import cv2
import numpy as np
import os

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64}

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ]

def R(val):
    return int(np.random.random()*val)

def rotRandrom(img,factor,shape):
        src_points=np.float32([[0,0],[0,shape[0]],[shape[1],0],[shape[1],shape[0]]])              #Original Four Point From input image
        dst_points=np.float32([[R(factor),R(factor)],[R(factor),shape[0]-R(factor)],
        [shape[1]-R(factor),R(factor)],[shape[1]-R(factor),shape[0]-R(factor)]])
        Transfer=cv2.getPerspectiveTransform(src_points,dst_points)
        output=cv2.warpPerspective(img,Transfer,shape)
        return output

def GaussBlur(img,level):
    img=cv2.GaussianBlur(img,(level*2+1,level*2+1),0)
    cv2.imshow("bg",img)
    cv2.waitKey()
    return img

def Add_Env(img,env_set):
    index=R(len(env_set))
    env=cv2.imread(env_set[index])
    env=cv2.resize(env,(img.shape[1],img.shape[0]))
    for i in range(0,img.shape[1]):
        for j in range(0,img.shape[0]):
            if img[j][i].any()==0:
                img[j][i]=env[j][i]
    return img



class GenPlate:
    def __init__(self,Zhttf,Enttf,Env_Path,flag):
        self.fontZh=ImageFont.truetype(Zhttf,43,0)
        self.fontEn=ImageFont.truetype(Enttf,60,0)
        self.img=np.array(Image.new("RGB",(226,70),(255,255,255)))
        self.bg  = cv2.resize(cv2.imread("/home/ly0kos/Car/images/template.bmp"),(226,70))
        self.smu = cv2.imread("/home/ly0kos/Car/images/smu2.jpg")
        self.env_path=[]
        for root,dir,filename in os.walk(Env_Path):
            for name in filename:
                path=os.path.join(root,name)
                self.env_path.append(path)
        self.save_flag=flag

    def GenZh(self,font,text):
        img=Image.new("RGB",(45,70),(255,255,255))
        draw=ImageDraw.Draw(img)
        draw.text((0,3),text,(0,0,0),font=font)
        img=img.resize((23,70))
        img=np.array(img)
        return img

    def GenEN(self,font,text):
        img=Image.new("RGB",(23,70),(255,255,255))
        draw=ImageDraw.Draw(img)
        draw.text((0,2),text,(0,0,0),font=font)
        img=np.array(img)
        return img

    def draw(self,text):
        offset= 2 
        self.img[0:70,offset+8:offset+8+23]= self.GenZh(self.fontZh,text[0])
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= self.GenEN(self.fontEn,text[1])
        for i in range(5):
            base = offset+8+23+6+23+17 +i*23 + i*6 
            self.img[0:70, base  : base+23]= self.GenEN(self.fontEn,text[i+2])
        return self.img

    def genStr(self):
        Str=""
        pos=0
        label=np.empty((7,1),dtype=np.int64)
        while (pos<7):
            if pos==0:
                temp=np.random.randint(0,31)
                Str += chars[temp]                                                                               #Chinese Char
                label[pos]=temp
                pos +=1
            elif pos==1:
                temp=np.random.randint(41,65)
                Str+=chars[temp]                                                                                 #EnglishAlphabet
                label[pos]=temp
                pos+=1
            else:
                temp=np.random.randint(31,41)
                Str+=chars[temp]
                label[pos]=temp
                pos+=1
        return label,Str

    def generate(self,text):
        plate=self.draw(text)
        plate=cv2.bitwise_not(plate)                                                                                    #黑底白字
        plate=cv2.bitwise_or(plate,self.bg)                                                                        #加入背景
        plate=rotRandrom(plate,15,(plate.shape[1],plate.shape[0]))
        plate=Add_Env(plate,self.env_path)
        plate=GaussBlur(plate,1+R(7))
        return plate
        

    def genBatch(self,batchSize,outputPath,size):
        data=[]
        label=np.empty((batchSize,7,1))
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        for i in range(batchSize):
            num,plateStr=self.genStr()
            img=self.generate(plateStr)
            img = cv2.resize(img,size)
            data.append(img)
            label[i]=num
            if self.save_flag==1:
                filename = os.path.join(outputPath, str(i).zfill(4) + '.' + plateStr + ".jpg")
                cv2.imwrite(filename, img)
        return data,label