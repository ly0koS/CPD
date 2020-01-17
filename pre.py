import os
import cv2


def getFileName(path):
    L=[]
    for root,dirs,files in os.walk(path):
        for name in files:
            L.append(name)
    num=len(os.listdir(path))
    return(L,num)


def pictureProcess(files,num):
    L=[]
    i=1
    while i<num:
        i+=1
        L.append("image%d" %i)
    i=0
    for j in files:
        L[i]=cv2.imread("/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_challenge/%s"%j)
        i+=1
pass
    

def main():
    files=[]
    num=0
    files,num=getFileName("/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_challenge")
    pictureProcess(files,num)

if __name__ == '__main__':
    main()