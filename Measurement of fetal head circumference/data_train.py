import numpy as np
import os
from PIL import Image
from until import *
import cv2
from model import *
import time
import matplotlib.pyplot as plt
#os.path.splitext()：分离文件名与扩展名
#os.listdir()：查找本目录下所有文件
train_data_path='D:/US data/training_set2/'
lenth = np.array(os.listdir(train_data_path))
Enchance_type=['ImgRotate','ImgLRMirror','ImgTBMirror','BrightEnchance','ColorEnchance','ContrastEnchance','SharpEnchance']
#np.zeros,np.ones(())
train_data=[]
train_Annotation=[]
#分开数据和注释
for i in lenth:
    if "Annotation" in i:
        train_Annotation.append(i)
    else:
        train_data.append(i)
#np.save('train_data.npy',train_data)
#np.save('train_Annotation.npy',train_Annotation)

def pre_process(path,Enchance_type,Enchance_value):
    img = Image.open(train_data_path+path)
    img=ImgResizeTo(img,(800,540))
    if Enchance_type=='ImgRotate':
        img=ImgRotate(img,Enchance_value)
        new_path=train_data_path+Enchance_type+'_'+str(Enchance_value)+'_'+path
        img.save(new_path)
    elif Enchance_type=='ImgLRMirror':
        img=ImgLRMirror(img)
        new_path =train_data_path+Enchance_type+'_'+path
        img.save(new_path)
    elif Enchance_type=='ImgTBMirror':
        img=ImgTBMirror(img)
        new_path =train_data_path+Enchance_type+'_'+path
        img.save(new_path)
    elif Enchance_type=='BrightEnchance':
        img=BrightEnchance(img,Enchance_value)
        new_path =train_data_path+Enchance_type+'_'+str(Enchance_value)+'_'+path
        img.save(new_path)
    elif Enchance_type=='ColorEnchance':
        img=ColorEnchance(img,Enchance_value)
        new_path =train_data_path+Enchance_type+'_'+str(Enchance_value)+'_'+path
        img.save(new_path)
    elif Enchance_type == 'ContrastEnchance':
        img = ContrastEnchance(img, Enchance_value)
        new_path =train_data_path + Enchance_type + '_' + str(Enchance_value) + '_' + path
        img.save(new_path)
    elif Enchance_type == 'SharpEnchance':
        img = SharpEnchance(img, Enchance_value)
        new_path =train_data_path + Enchance_type + '_' + str(Enchance_value) + '_' + path
        img.save(new_path)
    return new_path
train_Fill_Annotation=[]
#train_Enchance_data=[]
if not os.path.isfile(train_data_path+'train_Fil_Annotation.npy'):
    print('creat train_Fill_Annotation.npy')
    for n in range(len(train_data)):
        #填充
        im_out=FillHole(train_data_path+train_Annotation[n])
        # 保存结果
        cv2.imwrite(train_data_path+'Fill_'+train_Annotation[n], im_out)
        train_Fill_Annotation.append(train_data_path+'Fill_'+train_Annotation[n])
        np.save(train_data_path+'train_Fill_Annotation.npy',train_Fill_Annotation)
        #new_path =pre_process(train_data[n],Enchance_type[0],30)
        #train_Enchance_data.append(new_path)
else:
    print('load train_Fill_Annotation.npy')
    train_Fill_Annotation=np.load(train_data_path+'train_Fill_Annotation.npy')
epochs=100
batch=8
loss_sum=0
model=unet()
for epoch in range(epochs):
    tic = time.time()
    count = 0
    Indices = np.arange(len(train_data))  # 随机打乱索引
    np.random.shuffle(Indices)
    #train_y=np.empty((540,800,1))
    for times in range(int(len(train_Fill_Annotation)/batch)+1):
        tic = time.time()
        judge=0
        for id in Indices[batch*count:batch*(count+1)]:
            if train_data[id][0:3] in train_Fill_Annotation[id]:
                if judge==0:
                    train_x=read_data(train_data_path+train_data[id])
                    train_y=read_data(train_Fill_Annotation[id])
                    judge=1
                else:
                    train_x = np.concatenate((train_x, read_data(train_data_path+train_data[id])), axis=0)
                    train_y = np.concatenate((train_y, read_data(train_Fill_Annotation[id])),
                                             axis=0)
            else:
                print('Mismatch:'+train_data[id]+'and '+train_Fill_Annotation[id][24:])
        '''ima=train_x[0,:,:,0]
        cv2.imshow("Image", ima)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(train_y.shape)
        print(train_x.shape)
        ima=train_y[0,:,:,0]
        cv2.imshow("Image", ima)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(train_y.shape)
        print(train_x.shape)'''
        loss= model.train_on_batch(train_x,train_y)
        loss_sum += loss[0]
        toc = time.time()
        print("speed of progress %d/%d/%d finsh %.2f sec.loss=%.4f" % (count, int(len(train_Fill_Annotation)/batch)+1,epoch,toc - tic, loss[0]))
        count+=1
    toc = time.time()
    loss_sum = loss_sum / count
    print("epoch%f finsh %f sec.loss=%.4f" % (epoch, toc - tic, loss_sum))
    loss_sum = 0
    model.save('D:/python/US/model/model'+str(epoch) + '_net.h5')