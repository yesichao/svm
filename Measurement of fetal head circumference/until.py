import numpy as np
import os
import cv2
from PIL import Image, ImageEnhance
def read_data(path):
    img= Image.open(path)
    img = ImgResizeTo(img, (288, 224))
    img= np.array(img)
    #ex=np.zeros((800))
    #img=np.row_stack((ex,ex,img,ex,ex))#防止上采样后与池化大小不匹配
    #print(img.shape)
    img = img/ 255
    img=np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    return img
def read_shape(path):
    img= Image.open(path)
    img = np.array(img)
    return img.shape
def FillHole(imgPath,w=0,h=0):
    #im_in = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    im_in = Image.open(imgPath)
    if w!=0:
        im_in=np.array(ImgResizeTo(im_in,(w,h)))
    else:
        im_in = np.array(im_in)
    # 复制 im_in 图像
    im_floodfill = im_in.copy()
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv
    return im_floodfill_inv
def ImgResize(Img,ScaleFactor):
    ImgSize = Img.size #获得图像原始尺寸
    NewSize = [int(ImgSize[0]*ScaleFactor),int(ImgSize[1]*ScaleFactor)] #获得图像新尺寸，保持长宽比
    Img = Img.resize(NewSize)     #利用PIL的函数进行图像resize，类似matlab的imresize函数
    return Img

def ImgResizeTo(Img,NewSize):
    Img = Img.resize(NewSize)     #利用PIL的函数进行图像resize，类似matlab的imresize函数 WH
    return Img

#旋转
def ImgRotate(Img,Degree):
    return Img.rotate(Degree) #利用PIL的函数进行图像旋转，类似matlab imrotate函数

#利用PIL的函数进行水平以及上下镜像
def ImgLRMirror(Img):
    return Img.transpose(Image.FLIP_LEFT_RIGHT)

def ImgTBMirror(Img):
    return Img.transpose(Image.FLIP_TOP_BOTTOM)

# 亮度,增强因子为1.0是原始图像,增强因子为0.0将产生黑色图像
def BrightEnchance(Img, factor):
    enh_bri = ImageEnhance.Brightness(Img)
    image_brightened = enh_bri.enhance(factor)
    return image_brightened

# 色度,增强因子为1.0是原始图像,增强因子为0.0将产生黑白图像
def ColorEnchance(Img, factor):
    image_colored = ImageEnhance.Color(Img).enhance(factor)
    return image_colored

#对比度，增强因子为1.0是原始图片,增强因子为0.0将产生纯灰色图像
def ContrastEnchance(Img, factor):
    image_contrasted = ImageEnhance.Contrast(Img).enhance(factor)
    return image_contrasted

# 锐度，增强因子为1.0是原始图片,增强因子为0.0将产生模糊图像,为2.0将产生锐化过的图像
def SharpEnchance(Img, factor):
    image_sharped = ImageEnhance.Sharpness(Img).enhance(factor)
    return image_sharped