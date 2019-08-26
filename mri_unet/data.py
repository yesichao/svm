import numpy as np
import os
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 松弛一下限制
training_data_path_x = "D:/python/SpineSagT2Wdataset3/test/image/Case"
lenth = len(np.array(os.listdir('D:/python/SpineSagT2Wdataset3/test/image')))
x_test =[]
n=0
for i in range(196,lenth+195):
    n=n+1
    img_path_x = os.path.join(training_data_path_x+str(i)+'.nii.gz')
    img_x = nib.load(img_path_x).get_data()

    for i in range(img_x.shape[2]):   # 对切片进行循环
        img_2d_x= img_x[:, :, i]  # 取出一张图像
        img_2d_x = np.transpose(img_2d_x, (1, 0))
        if img_2d_x.shape[0]!=880:
            img_2d_x  = Image.fromarray(img_2d_x)
            img_2d_x = img_2d_x .resize((880,880), Image.ANTIALIAS)
            img_2d_x = np.matrix(img_2d_x.getdata(), dtype='float')
            img_2d_x = img_2d_x.reshape(880,880)
            mean = np.mean(img_2d_x)  # mean for data centering
            std = np.std(img_2d_x)  # std for data normalization
            img_2d_x -= mean
            img_2d_x /= std
            x_test.append(np.array(img_2d_x))
        else:
            mean = np.mean(img_2d_x)  # mean for data centering
            std = np.std(img_2d_x)  # std for data normalization
            img_2d_x -= mean
            img_2d_x /= std
            x_test.append(np.array(img_2d_x))
x_test = np.asarray(x_test, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
x_test = x_test[:, :, :, np.newaxis]  # 变成4维数据
print(x_test.shape)
print(np.max(x_test))
print(np.min(x_test))
print(n)
np.save('imgs_test.npy', x_test)
x_test=x_test.reshape(-1,880,880)
plt.imshow(x_test[3,:,:])
plt.show()

