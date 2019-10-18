import warnings
import wfdb
import numpy as np
from utils import *
import operator
data_path='D:/python/ecg/MIT-BIH'
DS1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
data = {'train': DS1,
        'test': DS2
        }
size_RR_max=20
target_class = ['train', 'test']
MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F','/','Q','f']
def creat_data():
    for i in range(len(target_class)):
        sig=[]
        sig_1=[]
        label=[]
        label_1 = []
        RR=[]
        RR_1 = []
        s=data[target_class[i]]
        for k in range(len(s)):
            print(data_path+'/'+ str(s[k])+target_class[i])
            record = wfdb.rdrecord(data_path + '/' + str(s[k]), sampfrom=0, channel_names=['MLII'])
            sigal1 = record.p_signal
            record = wfdb.rdrecord(data_path + '/' + str(s[k]), sampfrom=0, channels=[1])
            sigal2 = record.p_signal
            sigal = sigal1
            annotation = wfdb.rdann(data_path + '/' + str(s[k]), 'atr')
            for j in range(1,annotation.ann_len-2):
                pos=annotation.sample[j]
                if annotation.symbol[j] in MITBIH_classes:
                    if (pos > 90 and pos < (len(sigal) - 90)):
                        sign = sigal[pos - 90:pos + 90]
                        sign2 = sigal2[pos - 90:pos + 90]
                        if i==0:
                            RR.append((annotation.sample[j] - annotation.sample[j - 1],
                                       annotation.sample[j + 1] - annotation.sample[j]))
                            sig.append(sign)
                            if annotation.symbol[j] == 'N' or annotation.symbol[j] == 'L' or annotation.symbol[j] == 'R' or annotation.symbol[j] == 'e' or annotation.symbol[j] == 'j':
                                label.append(0)
                            elif annotation.symbol[j] == 'A' or annotation.symbol[j] == 'a' or annotation.symbol[j] == 'J' or annotation.symbol[j] == 'S':
                                label.append(1)
                            elif annotation.symbol[j] == 'V' or annotation.symbol[j] == 'E':
                                label.append(2)
                            elif annotation.symbol[j] == 'F':
                                label.append(3)
                            elif annotation.symbol[j] == '/' or annotation.symbol[j] == 'f' or annotation.symbol[j] == 'Q':
                                label.append(4)
                        else:
                            if annotation.sample[j]<=108000:
                                sig_1.append(sign)
                                RR_1.append((annotation.sample[j] - annotation.sample[j - 1],
                                           annotation.sample[j + 1] - annotation.sample[j]))
                                if annotation.symbol[j] == 'N' or annotation.symbol[j] == 'L' or annotation.symbol[
                                    j] == 'R' or annotation.symbol[j] == 'e' or annotation.symbol[j] == 'j':
                                    label_1.append(0)
                                elif annotation.symbol[j] == 'A' or annotation.symbol[j] == 'a' or annotation.symbol[
                                    j] == 'J' or annotation.symbol[j] == 'S':
                                    label_1.append(1)
                                elif annotation.symbol[j] == 'V' or annotation.symbol[j] == 'E':
                                    label_1.append(2)
                                elif annotation.symbol[j] == 'F':
                                    label_1.append(3)
                                elif annotation.symbol[j] == '/' or annotation.symbol[j] == 'f' or annotation.symbol[
                                    j] == 'Q':
                                    label_1.append(4)
                            else:
                                sig.append(sign)
                                RR.append((annotation.sample[j] - annotation.sample[j - 1],
                                           annotation.sample[j + 1] - annotation.sample[j]))
                                if annotation.symbol[j] == 'N' or annotation.symbol[j] == 'L' or annotation.symbol[
                                    j] == 'R' or annotation.symbol[j] == 'e' or annotation.symbol[j] == 'j':
                                    label.append(0)
                                elif annotation.symbol[j] == 'A' or annotation.symbol[j] == 'a' or annotation.symbol[
                                    j] == 'J' or annotation.symbol[j] == 'S':
                                    label.append(1)
                                elif annotation.symbol[j] == 'V' or annotation.symbol[j] == 'E':
                                    label.append(2)
                                elif annotation.symbol[j] == 'F':
                                    label.append(3)
                                elif annotation.symbol[j] == '/' or annotation.symbol[j] == 'f' or annotation.symbol[
                                    j] == 'Q':
                                    label.append(4)
        if i==1:
            print('saving 5 min test sample....')
            sig_1 = np.asarray(sig_1, dtype=np.float32)
            label_1=np.asarray(label_1, dtype=np.float32)
            RR_1 = np.asarray(RR_1, dtype=np.float32)
            np.save('D:/python/liangji/row_data/sig_5min_sig.npy', sig_1)
            np.save('D:/python/liangji/row_data/label_5min_sig.npy', label_1)
            np.save('D:/python/liangji/row_data/RR_5min_sig.npy', RR_1)
            print(sig_1.shape,label_1.shape,RR_1.shape)
        sig = np.asarray(sig, dtype=np.float32)
        label = np.asarray(label, dtype=np.float32)
        RR = np.asarray(RR, dtype=np.float32)
        np.save('D:/python/liangji/row_data/sig_' + target_class[i] + '_sig.npy', sig)
        np.save('D:/python/liangji/row_data/label_' + target_class[i] + '_sig.npy', label)
        np.save('D:/python/liangji/row_data/RR_' + target_class[i] + '_sig.npy', RR)
        print(sig.shape, label.shape,RR.shape)
creat_data()
