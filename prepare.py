from cv2 import cv2
import os

Path="/home/ly0kos/Car/chepai/images/"
OUTPUT="/home/ly0kos/Car/chepai/output/"

def find_end(start,end,white,black,arg,width,height):
    end=start+1
    for row in range(start+1,width-1):
        if(white(row)   if  arg else    black(row)>(height*0.95)):
            end=row
            break
    return end

def cut_char_from_plate(pic):
    out=os.path.join(OUTPUT,pic)
    pic=os.path.join(Path,pic)
    im = cv2.imread(pic)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)                                       # RGB2GRAY
    im = cv2.GaussianBlur(im, (3, 3), 0)                                                             # GaussianBlur
    im=cv2.resize(im, (480, 240))
    ret3, th3 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(out, th3)

    white = []
    black = []
    height = th3.shape[0]
    width = th3.shape[1]
    white_sum = 0                                                                                                   # WhitePixelSum
    black_sum = 0                                                                                                   # BlackPixelSum
    for i in range(width):
        w=0                                                                                                                     #WhitePixelInRow
        b=0                                                                                                                     #BlackPixelInRow
        for j in range(height):
            if th3[i][j] == 255:
                w += 1
            if th3[i][j] == 0:
                b += 1
        white_sum+=w
        black_sum+=b
        white.append(w)
        black.append(b)
        print(str(w)+"---------------"+str(b))
    if black_sum>white_sum:
        print("黑底白字")
        arg=False 
    else:
        print("白底黑字")
        arg=True
    
    row=1
    start=1
    end=1
    ###CutProvince###
    if ((black[row] if   arg    else   white[row])>(height*0.05)):                                                                #ThresholdForChar
        start=row
        end=find_end(start,white,black,arg,width,height)
        row=end
        if end-start>67:
            print("End - Start ="+str(end-start))
            province=th3[1:height,start:end]
            cv2.imshow("Province",province)
            cv2.waitKey(0)
