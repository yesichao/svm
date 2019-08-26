import warnings
import wfdb
import os
import numpy as np
import operator
from utils import *
from check_overtime import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
data_path='D:/python/bwl res_net/MIT-BIH'
DS1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
data = {'train': DS1,
        'test': DS2
        }
length=1440
target_class = ['train', 'test']
sign_class = ['N', 'S', 'V','F']
MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']
def creat_data():
    for i in range(len(target_class)):
        s=data[target_class[i]]
        if i==0:
            over_time = check_overtime(i,length)
        else:
            over_time = np.zeros(5)
        for sig_class in range(4):
            seg = []
            for k in range(len(s)):
                start_time = 0
                end_time = start_time + length
                print(data_path + '/' + str(s[k]) + target_class[i]+'  '+str(sig_class))
                record = wfdb.rdrecord(data_path + '/' + str(s[k]), sampfrom=0, channel_names=['MLII'])
                sigal = record.p_signal
                annotation = wfdb.rdann(data_path + '/' + str(s[k]), 'atr')
                while end_time <= sigal.shape[0]:
                    sign = sigal[start_time:end_time]
                    seg_class = []
                    for j in range(annotation.ann_len - 1):
                        if annotation.sample[j] >= start_time and annotation.sample[j] <= end_time:
                            if annotation.symbol[j] == 'N' or annotation.symbol[j] == 'L' or annotation.symbol[j] == 'R' or \
                                    annotation.symbol[j] == 'e' or annotation.symbol[j] == 'j':
                                seg_class.append(0)
                            elif annotation.symbol[j] == 'A' or annotation.symbol[j] == 'a' or annotation.symbol[j] == 'J' or annotation.symbol[j] == 'S':
                                seg_class.append(1)
                            elif annotation.symbol[j] == 'V' or annotation.symbol[j] == 'E':
                                seg_class.append(2)
                            elif annotation.symbol[j] == 'F':
                                seg_class.append(3)
                    if len(set(seg_class)) == 1 and seg_class[0] == 0:
                        seg_class_nb=0
                    elif len(set(seg_class)) != 0:
                        while 0 in seg_class:
                            seg_class.remove(0)
                        if max(seg_class, key=seg_class.count) == 1:
                            seg_class_nb = 1
                        elif max(seg_class, key=seg_class.count) == 2:
                            seg_class_nb = 2
                        elif max(seg_class, key=seg_class.count) == 3:
                            seg_class_nb = 3
                    else:
                        seg_class_nb = 4
                    if seg_class_nb==sig_class:
                        seg.append(sign)
                    start_time = end_time-int(over_time[sig_class])
                    end_time = start_time + length
            seg = np.asarray(seg, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
            np.save('D:/python/bwl res_net/row_data/'+sign_class[sig_class]+'_' + target_class[i] + '_seg.npy', seg)
            print(seg.shape)