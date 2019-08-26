import warnings
import wfdb
import time
from features_ECG import *
from utils import *
target_class = ['train', 'test']
target_names= ['N', 'S','V','F']
model_class = ['RR', 'HOS','lbp','Morph','wav']
model_svm_path='D:/python/svm/model_t/'
def creat_sig():
    DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
    DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    data = {'train': DS1,
            'test': DS2
            }
    sig_train = np.load('D:/python/svm/npy_180/data_' + target_class[0] + '_seg.npy')
    sig_test = np.load('D:/python/svm/npy_180/data_' + target_class[1] + '_seg.npy')
    for p in range(len(sig_train)):
        for n in range(len(sig_train[p])):
            data=sig_train[p][n][0]
            data=data.T
            if p==0 and n==0:
                x_train=data
            else:
                x_train = np.concatenate((x_train, data))
    print(x_train.shape)
    np.save('D:/python/svm/npy_180/sig_' + target_class[0] + '_seg.npy',x_train)
    for p in range(len(sig_test)):
        for n in range(len(sig_test[p])):
            data=sig_test[p][n][0]
            data=data.T
            if p==0 and n==0:
                x_test=data
            else:
                x_test= np.concatenate((x_test, data))
    print(x_test.shape)
    np.save('D:/python/svm/npy_180/sig_' + target_class[1] + '_seg.npy',x_test)
    return x_train,x_test
