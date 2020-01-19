import numpy as np
import cv2
from genplate import *

class PlateIter():
    def __init__(self,batch_size,height,width):
        self.genplate=GenPlate("./font/platech.ttf",'./font/platechar.ttf','./NoPlates')
        self.batch_size=batch_size
        self.height=height
        self.width=width
    def iter(self):
        data=[]
        