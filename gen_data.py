import numpy as np
import cv2
from genplate import *

def rand(lo,hi):
    return lo+r(hi-lo)

def gen_rand():
    name=""
    label=[]
    label.append(rand(0,31))                                                                                                                 #Gen Province
    label.append(rand(41,65))                                                                                                               #Second English Alphabet
    for i in range(5):
        label.append(rand_range(31,65))                                                                                             #3-8:Number
    name+=chars[label[0]]
    name+=chars[label[1]]
    for i in range(5):
         name+=chars(label[i+2])
    return name,label

def gen_plate(genplate,width,height):
    pass

class PlateIter():
    def __init__(self,batch_size,height,width):
        self.genplate=GenPlate("./font/platech.ttf",'./font/platechar.ttf','./NoPlates')
        self.batch_size=batch_size
        self.height=height
        self.width=width
    def iter(self):
        data=[]
        label=[]
        for i in range(self.batch_size):
            num,img=gen_plate()