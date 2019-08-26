import time
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from net_2 import *
import matplotlib.pyplot as plt
from ecg_seg import *
from utils import *
from keras.callbacks import Callback,ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score
N_sig=np.load('D:/python/bwl res_net/row_data/N_sig.npy').reshape(-1,1440)
S_sig=np.load('D:/python/bwl res_net/row_data/S_sig.npy').reshape(-1,1440)
V_sig=np.load('D:/python/bwl res_net/row_data/V_sig.npy').reshape(-1,1440)
F_sig=np.load('D:/python/bwl res_net/row_data/F_sig.npy').reshape(-1,1440)

N_sig_sample=np.load('D:/python/bwl res_net/row_data/N_sig_sample.npy')
S_sig_sample=np.load('D:/python/bwl res_net/row_data/S_sig_sample.npy')
V_sig_sample=np.load('D:/python/bwl res_net/row_data/V_sig_sample.npy')
F_sig_sample=np.load('D:/python/bwl res_net/row_data/F_sig_sample.npy')

N_sig_seg_class=np.load('D:/python/bwl res_net/row_data/N_sig_seg_class.npy')
S_sig_seg_class=np.load('D:/python/bwl res_net/row_data/S_sig_seg_class.npy')
V_sig_seg_class=np.load('D:/python/bwl res_net/row_data/V_sig_seg_class.npy')
F_sig_seg_class=np.load('D:/python/bwl res_net/row_data/F_sig_seg_class.npy')
k=8
sig=V_sig
sig_sample=V_sig_sample
seg_class=V_sig_seg_class
plt.plot(sig[k,:],'y',label='V')
print(sig_sample[k])
print(seg_class[k])
for i in range(len(sig_sample[k])):
    x=sig_sample[k][i]
    y=sig[k,x]
    if seg_class[k][i]==0:
        label='N'
    elif seg_class[k][i]==1:
        label='S'
    elif seg_class[k][i]==2:
        label='V'
    elif seg_class[k][i]==3:
        label='F'
    elif seg_class[k][i]==4:
        label='Q'
    plt.annotate(label, xy=(x,y))

plt.legend()
plt.savefig('4.png')
plt.show()