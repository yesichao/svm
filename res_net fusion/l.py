import warnings
import wfdb
import os
import numpy as np
import operator
from utils import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
data_path='D:/python/bwl res_net/MIT-BIH'
DS1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
data = {'train': DS1,
        'test': DS2
        }
target_class = ['train', 'test']
MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']
def check_overtime(i,length):
    s = data[target_class[i]]
    N_sig = []
    S_sig = []
    V_sig = []
    F_sig = []
    N_sig_sample = []
    S_sig_sample = []
    V_sig_sample = []
    F_sig_sample = []
    N_sig_seg_class = []
    S_sig_seg_class = []
    V_sig_seg_class = []
    F_sig_seg_class = []

    over_time = np.zeros(4)
    for k in range(len(s)):
        start_time = 0
        end_time = start_time + length
        print(data_path + '/' + str(s[k]) + target_class[i])
        record = wfdb.rdrecord(data_path + '/' + str(s[k]), sampfrom=0, channel_names=['MLII'])
        sigal = record.p_signal
        annotation = wfdb.rdann(data_path + '/' + str(s[k]), 'atr')
        while end_time <= sigal.shape[0]:
            sign = sigal[start_time:end_time]
            seg_class = []
            seg_class1 = []
            sample=[]
            for j in range(annotation.ann_len - 1):
                if annotation.sample[j] >= start_time and annotation.sample[j] <= end_time:
                    sample.append(annotation.sample[j]-start_time)
                    if annotation.symbol[j] == 'N' or annotation.symbol[j] == 'L' or annotation.symbol[j] == 'R' or \
                            annotation.symbol[j] == 'e' or annotation.symbol[j] == 'j':
                        seg_class.append(0)
                        seg_class1.append(0)
                    elif annotation.symbol[j] == 'A' or annotation.symbol[j] == 'a' or annotation.symbol[j] == 'J' or \
                            annotation.symbol[j] == 'S':
                        seg_class.append(1)
                        seg_class1.append(1)
                    elif annotation.symbol[j] == 'V' or annotation.symbol[j] == 'E':
                        seg_class.append(2)
                        seg_class1.append(2)
                    elif annotation.symbol[j] == 'F':
                        seg_class.append(3)
                        seg_class1.append(3)
                    elif annotation.symbol[j] == '/' or annotation.symbol[j] == 'f' or annotation.symbol[j] == 'Q':
                        seg_class.append(4)
                        seg_class1.append(4)
            if len(set(seg_class)) == 1 and seg_class[0] == 0:
                N_sig.append(sign)
                N_sig_sample.append(sample)
                N_sig_seg_class.append(seg_class1)
            elif len(set(seg_class)) != 0:
                while 0 in seg_class:

                    seg_class.remove(0)
                if max(seg_class, key=seg_class.count) == 1:
                    S_sig.append(sign)
                    S_sig_sample.append(sample)
                    S_sig_seg_class.append(seg_class1)
                elif max(seg_class, key=seg_class.count) == 2:
                    V_sig.append(sign)
                    V_sig_sample.append(sample)
                    V_sig_seg_class.append(seg_class1)
                elif max(seg_class, key=seg_class.count) == 3:
                    F_sig.append(sign)
                    F_sig_sample.append(sample)
                    F_sig_seg_class.append(seg_class1)

            start_time = end_time
            end_time = start_time + length
    num = [len(N_sig), len(S_sig), len(V_sig), len(F_sig)]
    print([len(N_sig_sample), len(S_sig_sample), len(V_sig_sample), len(F_sig_sample)])
    print([len(N_sig_seg_class), len(S_sig_seg_class), len(V_sig_seg_class), len(F_sig_seg_class)])
    print([len(N_sig), len(S_sig), len(V_sig), len(F_sig)])
    N_sig = np.asarray(N_sig, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
    S_sig = np.asarray(S_sig, dtype=np.float32)
    V_sig = np.asarray(V_sig, dtype=np.float32)
    F_sig = np.asarray(F_sig, dtype=np.float32)
    np.save('D:/python/bwl res_net/row_data/N_sig.npy',N_sig)
    np.save('D:/python/bwl res_net/row_data/S_sig.npy', S_sig)
    np.save('D:/python/bwl res_net/row_data/V_sig.npy', V_sig)
    np.save('D:/python/bwl res_net/row_data/F_sig.npy', F_sig)

    np.save('D:/python/bwl res_net/row_data/N_sig_sample.npy',N_sig_sample)
    np.save('D:/python/bwl res_net/row_data/S_sig_sample.npy', S_sig_sample)
    np.save('D:/python/bwl res_net/row_data/V_sig_sample.npy', V_sig_sample)
    np.save('D:/python/bwl res_net/row_data/F_sig_sample.npy', F_sig_sample)

    np.save('D:/python/bwl res_net/row_data/N_sig_seg_class.npy',N_sig_seg_class)
    np.save('D:/python/bwl res_net/row_data/S_sig_seg_class.npy', S_sig_seg_class)
    np.save('D:/python/bwl res_net/row_data/V_sig_seg_class.npy', V_sig_seg_class)
    np.save('D:/python/bwl res_net/row_data/F_sig_seg_class.npy', F_sig_seg_class)
    return num
check_overtime(0,1440)
