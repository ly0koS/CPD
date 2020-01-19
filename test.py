import os
from cv2 import cv2


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
        L[i]=cv2.imread("/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_base/%s"%j)                #Read File
        L[i]=cv2.cvtColor(L[i],cv2.COLOR_RGB2GRAY)                                                                                          #RGB2GRAY
        L[i]=cv2.GaussianBlur(L[i],(3,3),0)                                                                                                                  #GaussianBlur
        L[i]=cv2.resize(L[i],(128,128))
        cv2.imwrite("output/%s.jpg"%i,L[i])                                                                                                            #Save File
        i+=1
        if i>50:
            break
        
    
    
    

def main():
    files=[]
    num=0
    if not os.path.exists("output"):
        os.mkdir("output")

    files,num=getFileName("/home/ly0kos/WD/tensorflow/ccpd_dataset/ccpd_base")
    print(len(files))
    pictureProcess(files,num)

if __name__ == '__main__':
    main()
