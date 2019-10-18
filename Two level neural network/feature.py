import warnings
import wfdb
import numpy as np
from utils import *
import operator
import math
target_class = ['train', 'test','5min']
def z_score(data):
    for i in range(data.shape[0]):
        data[i]=(data[i]-np.mean(data))/np.std(data, ddof = 1)
    return data
def kurtosis(data):
    data=data.reshape(data.shape[0],data.shape[1])
    kur=np.array([], dtype=np.float32)
    for i in range (data.shape[0]):
        kur_mid = np.array([], dtype=np.float32)
        if i%1000==0:
            print('calculate kurtosis on %d/%d....'%(i,data.shape[0]))
        for j in range(data.shape[1]):
            kur_mid=np.append(kur_mid,math.pow((data[i,j]-np.mean(data[i,:])),4))
        kur=np.append(kur,np.mean(kur_mid))
    return z_score(kur)
def skewness(data):
    data=data.reshape(data.shape[0],data.shape[1])
    ske=np.array([], dtype=np.float32)
    for i in range (data.shape[0]):
        ske_mid = np.array([], dtype=np.float32)
        if i%1000==0:
            print('calculate skewness on %d/%d....'%(i,data.shape[0]))
        for j in range(data.shape[1]):
            ske_mid=np.append(ske_mid,math.pow((data[i,j]-np.mean(data[i,:])),4))
        ske=np.append(ske,np.mean(ske_mid)/math.pow((np.std(data[i,:], ddof = 1)),3))
    return z_score(ske)
def sample_range(data):
    data = data.reshape(data.shape[0], data.shape[1])
    return z_score(np.max(data,axis=1)-np.min(data,axis=1))
def sig_wt_component(data):
    data = data.reshape(data.shape[0], data.shape[1])
    A4=[]
    D2 = []
    D3 = []
    D4 =[]
    for i in range(data.shape[0]):
        if i%1000==0:
            print('calculate sig_wt_component on %d/%d....'%(i,data.shape[0]))
        sig = data[i,:]
        coeffs = pywt.wavedec(sig, 'db4', level=4)
        A4.append(pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0]).tolist(), 'db4'))
        D2.append(pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0]).tolist(), 'db4'))
        D3.append(pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0]).tolist(), 'db4'))
        D4.append(pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0]).tolist(), 'db4'))
    A4 = np.asarray(A4, dtype=np.float32)
    D2 = np.asarray(D2, dtype=np.float32)
    D3 = np.asarray(D3, dtype=np.float32)
    D4 = np.asarray(D4, dtype=np.float32)
    return A4,D2,D3,D4
for i in range(len(target_class)):
    sig=np.load('D:/python/liangji/row_data/sig_' + target_class[i] + '_sig.npy')
    print('creating '+ target_class[i] + ' mix feature....')
    P_kur = kurtosis(sig[:, 0:68, :]).reshape(-1,1)
    print(P_kur.shape)
    P_ske = skewness(sig[:, 0:68, :]).reshape(-1,1)
    P_range=sample_range(sig[:, 0:68, :]).reshape(-1,1)
    QRS_kur=kurtosis(sig[:, 68:112, :]).reshape(-1,1)
    QRS_ske = skewness(sig[:, 68:112, :]).reshape(-1,1)
    T_kur=kurtosis(sig[:, 112:180, :]).reshape(-1,1)
    T_ske = skewness(sig[:, 112:180, :]).reshape(-1,1)
    print(np.concatenate((P_kur,P_ske,P_range,QRS_kur,QRS_ske,T_kur,T_ske),axis=1).shape)
    np.save('D:/python/liangji/feature/mix_'+ target_class[i] + '.npy',np.concatenate((P_kur,P_ske,P_range,QRS_kur,QRS_ske,T_kur,T_ske),axis=1))
    print('creating ' + target_class[i] + ' sig_wav feature....')
    A4,D2,D3,D4=sig_wt_component(sig[:,68:112,:])
    print(A4.shape,D2.shape,D3.shape,D4.shape)
    np.save('D:/python/liangji/feature/A4_' + target_class[i] + '.npy',A4)
    np.save('D:/python/liangji/feature/D2_' + target_class[i] + '.npy', D2)
    np.save('D:/python/liangji/feature/D3_' + target_class[i] + '.npy', D3)
    np.save('D:/python/liangji/feature/D4_' + target_class[i] + '.npy', D4)


