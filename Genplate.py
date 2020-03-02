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

class GenPlate():
    def __init__(self,Zhttf,Enttf):
        self.fontZh=ImageFont.truetype(Zhttf,43,0)
        self.fontEn=ImageFont.truetype(Enttf,60,0)
        self.img=np.array(Image.new("RGB",(226,70),(255,255,255)))
        self.bg  = cv2.resize(cv2.imread("/home/ly0kos/Car/images/template.bmp"),(226,70))
        self.smu = cv2.imread("/home/ly0kos/Car/images/smu2.jpg")
        
    def draw(self,val):
        pass

    def genStr(self):
        Str=""
        pos=0
        while (pos<7):
            if pos==0:
                Str += chars[np.randint(0,31)]                                      #Chinese Char
                pos +=1
            elif pos==1:
                Str+=chars[np.randint(41,65)]                                      #EnglishAlphabet
                pos+=1
            else:
                Str+=chars[np.randint(31,41)]
                pos+=1
        return Str

    def genBatch(self,batchSize,outputPath,size):
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        for i in range(batchSize):
            plateStr=self.genStr()
            #TODO:GenerateImg