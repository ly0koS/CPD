import os
from cv2 import cv2
from prepare import get_test_data,get_train_data

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
    path="/home/ly0kos/Car/plate/images"
    if not os.path.exists("output"):
        os.mkdir("../output")

    files,num=getFileName(path)
    print(len(files))
    
    for i in files:
        get_train_data(i)

if __name__ == '__main__':

    main()
