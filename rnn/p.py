import time
import numpy as np
import h5py as hp
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.layers import Add
from keras.layers import MaxPooling1D
from keras.layers.core import Lambda
from keras.models import Model
from keras import regularizers
from keras.models import Input
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.layers import Dropout,Flatten
from keras.layers.core import Dense, Activation
from keras.layers import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.layers import LSTM
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr
Path='D:/python/cnn/' #自定义路径要正确
DataFile='Data_CNN.mat'
LabelFile='Label_OneHot.mat'

print("Loading data and labels...")
tic=time.time()
Data=load_mat(Path+DataFile,'Data')
Label=load_mat(Path+LabelFile,'Label')
Data=Data.T
Indices=np.arange(Data.shape[0]) #随机打乱索引并切分训练集与测试集
np.random.shuffle(Indices)

print("Divide training and testing set...")
train_x=Data[Indices[:10000]]
train_y=Label[Indices[:10000]]
test_x=Data[Indices[10000:]]
test_y=Label[Indices[10000:]]
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")
print(train_x.shape)
