import os
import warnings
import numpy as np
import time
import keras
from keras.callbacks import ModelCheckpoint
from mri_model import unet,dice_coef_np
from keras.callbacks import TensorBoard
from mri_unetadd import Nest_Net
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
print("Loading data and labels...")
tic=time.time()
x_train=np.load("D:/python/imgs_train.npy")
y_train=np.load("D:/python/imgs_mask_train.npy")
x_val=np.load("D:/python/imgs_val.npy")
y_val=np.load("D:/python/imgs_mask_val.npy")
print(x_train.shape[0])
Indices=np.arange(x_train.shape[0]) #随机打乱索引
np.random.shuffle(Indices)
Indices=Indices[:x_train.shape[0]]
print(Indices.shape)
x_train=x_train[Indices,:,:,:]
y_train=y_train[Indices,:,:,:]
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")
batch_size = 2
epochs = 30
print("2D-unet setup and initialize...")
tic=time.time()
model=unet('D:/python/model_t/unet_model_2.hdf5')
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")
print("2D-unet training and testing...")
tic=time.time()
MODEL_PATH = 'D:/python/model_t/'
model_name = 'unet_model_' + str(5) + '.hdf5'
checkpoint = ModelCheckpoint(filepath=MODEL_PATH+model_name,
                             monitor='loss', mode='min',
                             save_best_only='True')
tbCallBack = keras.callbacks.TensorBoard(log_dir='D:/python/10s/model_t/tmp/log',
                                         histogram_freq=1,
                                         write_graph=True,
                                         write_images=True)
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_val,y_val),callbacks=[tbCallBack,checkpoint])
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")