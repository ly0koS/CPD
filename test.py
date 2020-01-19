import os
from cv2 import cv2


def getFileName(path):
    L=[]
    for root,dirs,files in os.walk(path):
        for name in files:
            L.append(name)
    num=len(os.listdir(path))
    return(L,num)


def pictureProcess(files,num,path):
    L=[]
    i=0
    while i<num:
        i+=1
        L.append("image%d" %i)
    i=0 
    for j in files:
        if i<50:
            dir=path+"%s"%j
            L[i]=cv2.imread(dir)                #Read File
            L[i]=cv2.cvtColor(L[i],cv2.COLOR_RGB2GRAY)                                                                                          #RGB2GRAY
            L[i]=cv2.GaussianBlur(L[i],(3,3),0)                                                                                                                  #GaussianBlur
            L[i]=cv2.resize(L[i],(480,480))
            cv2.imwrite("output/%s.jpg"%i,L[i])                                                                                                            #Save File
        i+=1
        

def main():
    files=[]
    num=0
    path="/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_base/"
    if not os.path.exists("output"):
        os.mkdir("output")

    files,num=getFileName(path)
    print(len(files))
    pictureProcess(files,num,path)

if __name__ == '__main__':
    main()
