import warnings
import numpy as np
import time
import cv2
import math
from keras.models import Model, load_model
from model import dice_coef_loss,dice_coef
from until import *
import xlrd
import csv
from pandas import read_csv
excel_data=[]
with open('D:/US data/test_set_pixel_size.csv','r',encoding='utf-8') as f:
    reader=csv.reader(f)
    for row in reader:  # 将csv 文件中的数据保存到excel_data中
        excel_data.append(row)
    f.close()

res=["head circumference (mm)"]

def Perimeter(b,a,pix_size):#椭圆周长计算
    return (2*math.pi*b+4*(a-b))*pix_size
test_data_path='D:/US data/test_set/'
test_numb=os.listdir(test_data_path)
save=0
for i in range(len(test_numb)):
    img_row = Image.open(test_data_path + test_numb[i])
    img_row = np.array(img_row)
    row_shape = read_shape(test_data_path + test_numb[i])
    im = np.zeros((row_shape[0], row_shape[1]))
    img=FillHole('D:/python/US/result/pre_'+test_numb[i],row_shape[1], row_shape[0])
    contours, _=cv2.findContours(img, cv2.RETR_LIST,  cv2.CHAIN_APPROX_NONE)#轮廓提取
    th = 0
    for s in contours:
        if th <= cv2.contourArea(s):
            th = cv2.contourArea(s)
            cnt=s
    ellipse = cv2.fitEllipse(cnt)  # 椭圆拟合
    new_im = cv2.ellipse(im, ellipse, 255, 2)
    new_im2 = cv2.ellipse(img_row, ellipse, 255, 2)
    if save!=0:
        print('saving D:/python/US/result/' + test_numb[i])
        cv2.imwrite('D:/python/US/result/' + test_numb[i], new_im)
        cv2.imwrite('D:/python/US/result/res_' + test_numb[i], new_im2)
    if excel_data[i+1][0][0:3] in test_numb[i]:
        res.append(Perimeter(ellipse[1][0]/2,ellipse[1][1]/2,float(excel_data[i+1][1])))
    else:
        print('Mismatch:'+excel_data[i+1][0]+' and '+test_numb[i])
        res.append(['Mismatch:'+excel_data[i+1][0]+' and '+test_numb[i]])
print(res)
'''
    print(ellipse[1][0],ellipse[1][1])
    for j in range(len(contours)):
        area = cv2.contourArea(contours[j])
        if area < th:
            img = cv2.drawContours(img, [contours[j]], 0, 0, -1)
    img = cv2.drawContours(img, cnt, 3, 255, 3)
    im = cv2.ellipse(im, ellipse, (0, 255, 0), 2)
    perimeter = cv2.arcLength(cnt,True)#周长计算    
    cv2.imshow(test_numb[i],new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''