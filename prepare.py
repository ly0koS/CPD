from cv2 import cv2

def  cut_char_from_plate(pic):
    im=cv2.imread(pic)
    im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)                                                                              #RGB2GRAY
    im=cv2.GaussianBlur(im,(3,3),0)                                                                                                      #GaussianBlur
    ret3,th3=cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.resize(im,(480,480))
    cv2.imwrite("%s"pic,th3)

    white=[]
    black=[]
    height=th3.shape[0]
    width=th3.shape[1]
    