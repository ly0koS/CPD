import os
from cv2 import cv2
from prepare import *

def getFileName(path):
    L=[]
    for root,dirs,files in os.walk(path):
        for name in files:
            L.append(name)
    num=len(os.listdir(path))
    return(L,num)


        

def main():
    files=[]
    num=0
    path="/home/ly0kos/Car/chepai/images"
    if not os.path.exists("output"):
        os.mkdir("../output")

    files,num=getFileName(path)
    print(len(files))
    
    for i in files:
        cut_char_from_plate(i)
        break

if __name__ == '__main__':
    main()
