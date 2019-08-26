import numpy as np
import os
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 松弛一下限制
training_data_path_x = "D:/python/mri/SpineSagT2Wdataset3/train/image/Case"
training_data_path_y="D:/python/mri/SpineSagT2Wdataset3/train/groundtruth/mask_case"
lenth = len(np.array(os.listdir('D:/python/mri/SpineSagT2Wdataset3/train/image')))
x_train =[]
y_train=[]
n=0
for i in range(2,lenth+1-30):
    n=n+1
    img_path_x = os.path.join(training_data_path_x+str(i)+'.nii.gz')
    img_x = nib.load(img_path_x).get_data()
    img_path_y = os.path.join(training_data_path_y+str(i)+'.nii.gz')
    img_y= nib.load(img_path_y).get_data()
    for i in range(img_x.shape[2]):   # 对切片进行循环
        img_2d_x= img_x[:, :, i]  # 取出一张图像
        img_2d_y = img_y[:, :, i]  # 取出一张图像
        img_2d_x = np.transpose(img_2d_x, (1, 0))
        img_2d_y = np.transpose(img_2d_y, (1, 0))
        if img_2d_x.shape[0]!=880:
            img_2d_x  = Image.fromarray(img_2d_x)
            img_2d_x = img_2d_x .resize((880,880), Image.ANTIALIAS)
            img_2d_x = np.matrix(img_2d_x.getdata(), dtype='float')
            img_2d_x = img_2d_x.reshape(880,880)
            img_2d_y = img_2d_y*0.9
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
            x_train.append(np.array(img_2d_x))
            y_train.append(np.array(img_2d_y))
        else:
            mean = np.mean(img_2d_x)  # mean for data centering
            std = np.std(img_2d_x)  # std for data normalization
            img_2d_x -= mean
            img_2d_x /= std
            x_train.append(np.array(img_2d_x))
            y_train.append(np.array(img_2d_y))
x_train = np.asarray(x_train, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
x_train = x_train[:, :, :, np.newaxis]  # 变成4维数据
y_train = np.asarray(y_train, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
y_train = y_train[:, :, :, np.newaxis]  # 变成4维数据
print(y_train.shape)
print(x_train.shape)
print(np.max(y_train))
print(np.min(y_train))
print(n)
np.save('imgs_train.npy', x_train)
np.save('imgs_mask_train.npy', y_train)
x_val =[]
y_val=[]
n=0
for i in range(lenth+1-30,lenth+1):
    n=n+1
    img_path_x = os.path.join(training_data_path_x+str(i)+'.nii.gz')
    img_x = nib.load(img_path_x).get_data()
    img_path_y = os.path.join(training_data_path_y+str(i)+'.nii.gz')
    img_y= nib.load(img_path_y).get_data()
    for i in range(img_x.shape[2]):   # 对切片进行循环
        img_2d_x= img_x[:, :, i]  # 取出一张图像
        img_2d_y = img_y[:, :, i]  # 取出一张图像
        img_2d_x = np.transpose(img_2d_x, (1, 0))
        img_2d_y = np.transpose(img_2d_y, (1, 0))
        if img_2d_x.shape[0]!=880:
            img_2d_x  = Image.fromarray(img_2d_x)
            img_2d_x = img_2d_x .resize((880,880), Image.ANTIALIAS)
            img_2d_x = np.matrix(img_2d_x.getdata(), dtype='float')
            img_2d_x = img_2d_x.reshape(880,880)
            img_2d_y = img_2d_y*0.9
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
            x_val.append(np.array(img_2d_x))
            y_val.append(np.array(img_2d_y))
        else:
            mean = np.mean(img_2d_x)  # mean for data centering
            std = np.std(img_2d_x)  # std for data normalization
            img_2d_x -= mean
            img_2d_x /= std
            x_val.append(np.array(img_2d_x))
            y_val.append(np.array(img_2d_y))
x_val = np.asarray(x_val, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
x_val = x_val[:, :, :, np.newaxis]  # 变成4维数据
y_val = np.asarray(y_val, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
y_val = y_val[:, :, :, np.newaxis]  # 变成4维数据
print(y_val.shape)
print(x_val.shape)
print(np.max(y_train))
print(np.min(y_train))
print(n)
np.save('imgs_val.npy', x_val)
np.save('imgs_mask_val.npy', y_val)
x_train=x_train.reshape(-1,880,880)
plt.imshow(x_train[3,:,:])
plt.show()
x_val=x_val.reshape(-1,880,880)
plt.imshow(x_val[3,:,:])
plt.show()