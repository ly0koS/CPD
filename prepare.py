from cv2 import cv2
import os
import gc


Path = "/home/ly0kos/Car/plate/images/"
HOME = "/home/ly0kos/Car/plate/"
city = ['京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕',
        '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼']

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
'V', 'W', 'X', 'Y', 'Z']

def find_end(start, white, black, white_max, black_max, arg, width, height):
    end = start + 1
    m = end
    print("Start at pixel = ",start)
    while m-start < 90:
        if (white[m] if arg else black[m]) > (white_max * 0.9 if arg else 0.9 * black_max):
            if m-start < 72:
                m+=1
            elif ((black[m] if arg else white[m]) > (black_max * 0.15 if arg else 0.15 * white_max)):
                m+=1
            else:
                end = m
                break
        else:
            m+=1
    end=m
    print("End at piexl = ",end)
    return end


def cut_char_from_plate(pic):
    picture = os.path.join(Path, pic)
    im = cv2.imread(picture)
    # RGB2GRAY
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # GaussianBlur
    im = cv2.GaussianBlur(im, (3, 3), 0)
    im = cv2.resize(im, (480, 240))
    ret, th3 = cv2.threshold(im, 125, 255, cv2.THRESH_BINARY)

    white = []
    black = []
    height = th3.shape[0]
    width = th3.shape[1]
    # WhitePixelSum
    white_max = 0
    # BlackPixelSum
    black_max = 0
    for i in range(width):
        w = 0  # WhitePixelInRow
        b = 0  # BlackPixelInRow
        for j in range(height):
            if th3[j][i] == 255:
                w += 1
            if th3[j][i] == 0:
                b += 1
        white_max = max(white_max, w)
        black_max = max(black_max, b)
        white.append(w)
        black.append(b)
    if black_max > white_max:
        #print("黑底白字")
        arg = False
    else:
        #print("白底黑字")
        arg = True

    row = 1
    start = 1
    end = 1
    print(pic)
    ###CutProvince###
    while row < 80:
        row += 1
        if ((black[row] if arg else white[row]) > (black_max * 0.1 if arg else 0.1 * white_max)):  # ThresholdForChar
            start = row
            end = find_end(start=start, white=white, black=black, white_max=white_max,
                           black_max=black_max, arg=arg, width=width, height=height)
            row = end
            if end-start >= 65:
                print("Province_Length = "+str(end-start))
                province = th3[1:height, start:end]
                return th3, province, height, width, end

def get_province(path,pic):
    

    th3, province, height, width, end = cut_char_from_plate(pic)
    
    name = pic[0:1]
    name = os.path.join(path, name)
    name = os.path.join(name, pic)
    cv2.imwrite(name, province)
    return end

def get_char(start,path):
    pass

def get_test_data(pic):

    pass


def get_train_data(pic):
    for i in city:
        cityfolder = os.path.join(HOME, "train/city", i)
        if not os.path.exists(cityfolder):
            os.makedirs(cityfolder)
    for i in ALPHABET:
        folder=os.path.join(HOME, "train/rest", i)
        if not os.path.exists(folder):
            os.makedirs(folder)
    for i in number:
        folder=os.path.join(HOME, "train/rest", i)
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    city_path=os.path.join(HOME, "train/city")
    char_path=os.path.join(HOME, "train/rest")

    start=get_province(city_path,pic)

    get_char(start,char_path,pic)
    
