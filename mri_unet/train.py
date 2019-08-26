import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import time
import keras
from keras.callbacks import ModelCheckpoint
from mri_model import unet,dice_coef_np
from keras.callbacks import TensorBoard
from mri_unetadd import Nest_Net
nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 松弛一下限制
training_data_path_x = "D:/python/mri/SpineSagT2Wdataset3/train/image/Case"
training_data_path_y="D:/python/mri/SpineSagT2Wdataset3/train/groundtruth/mask_case"
lenth = len(np.array(os.listdir('D:/python/mri/SpineSagT2Wdataset3/train/image')))
batch_x =[]
batch_y=[]
n=0
k=0
loss_sum=0
num=0
batch=2
epoch=3
model=unet()
Indices=np.arange(165) #随机打乱索引
np.random.shuffle(Indices)
Indices=Indices[2:164]
print(Indices.shape)
for l in range(1,epoch+1):
    print('training data epoch%f........'%(l))
    tic = time.time()
    for i in Indices:
        if i!=0 and i!=1:
            img_path_x = os.path.join(training_data_path_x+str(i)+'.nii.gz')
            img_x = nib.load(img_path_x).get_data()
            img_path_y = os.path.join(training_data_path_y+str(i)+'.nii.gz')
            img_y= nib.load(img_path_y).get_data()
            Indices1 = np.arange(img_x.shape[2])  # 随机打乱索引
            np.random.shuffle(Indices1)
            Indices1 = Indices1[:img_x.shape[2]]
            for j in Indices1:   # 对切片进行循环
                img_2d_x= img_x[:, :, j]  # 取出一张图像
                img_2d_y = img_y[:, :, j]  # 取出一张图像
                img_2d_x = np.transpose(img_2d_x, (1, 0))
                img_2d_y = np.transpose(img_2d_y, (1, 0))
                if img_2d_x.shape[0]!=880:
                    img_2d_x  = Image.fromarray(img_2d_x)
                    img_2d_x = img_2d_x .resize((880,880), Image.ANTIALIAS)
                    img_2d_x = np.matrix(img_2d_x.getdata(), dtype='float')
                    img_2d_x = img_2d_x.reshape(880,880)
                    img_2d_y = img_2d_y*0.99
                    img_2d_y  = Image.fromarray(img_2d_y)
                    img_2d_y = img_2d_y .resize((880,880), Image.ANTIALIAS)
                    img_2d_y = np.matrix(img_2d_y.getdata(), dtype='float')
                    img_2d_y = img_2d_y.reshape(880,880)
                    img_2d_y[img_2d_y > 0.5] = 1
                    img_2d_y[img_2d_y <= 0.5] = 0
                    mean = np.mean(img_2d_x)  # mean for data centering
                    std = np.std(img_2d_x)  # std for data normalization
                    img_2d_x -= mean
                    img_2d_x /= std
                    batch_x.append(np.array(img_2d_x))
                    batch_y.append(np.array(img_2d_y))
                    n=n+1
                else:
                    mean = np.mean(img_2d_x)  # mean for data centering
                    std = np.std(img_2d_x)  # std for data normalization
                    img_2d_x -= mean
                    img_2d_x /= std
                    batch_x.append(np.array(img_2d_x))
                    batch_y.append(np.array(img_2d_y))
                    n=n+1
                if n==batch:
                    tic = time.time()
                    batch_x = np.asarray(batch_x, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
                    batch_x = batch_x[:, :, :, np.newaxis]  # 变成4维数据
                    batch_y = np.asarray(batch_y, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
                    batch_y = batch_y[:, :, :, np.newaxis]  # 变成4维数据
                    loss=model.train_on_batch(batch_x, batch_y)
                    loss_sum += loss
                    num=num+1
                    toc = time.time()
                    if k<=Indices1.shape[0]:
                        print('traing on ' + training_data_path_y + str(i) + '.nii.gz',
                              '---' + str(k)+'/'+str(Indices1.shape[0])+'--train_loss = %.4f'%(loss)+" time is %f sec." % (toc - tic))
                    else:
                        k=0
                        print('traing on ' + training_data_path_y + str(i) + '.nii.gz',
                              '---' + str(k) + '/' + str(Indices1.shape[0])+'--train_loss = %.4f'%(loss)+" time is %f sec." % (toc - tic))
                    n=0
                    k = k + batch
                    batch_x=[]
                    batch_y=[]
                else:
                    continue
        else:
            continue
        toc = time.time()
        loss_sum=loss_sum/num
        print("epoch%f finsh %f sec.loss=%.4f"% (l,toc - tic,loss_sum))
        loss_sum=0
        num=0