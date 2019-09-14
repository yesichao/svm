import warnings
import wfdb
import time
from utils import *
import math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
target_sig_length=3600
warnings.filterwarnings("ignore")
data_path='D:/python/ecg/MIT-BIH'
s=get_image_files(data_path)
N_sig=[]
S_sig=[]
V_sig=[]
F_sig=[]
Q_sig=[]
N_sig_1=[]
S_sig_1=[]
V_sig_1=[]
F_sig_1=[]
Q_sig_1=[]
N_sig_2=[]
S_sig_2=[]
V_sig_2=[]
F_sig_2=[]
Q_sig_2=[]
N_rrd=[]
S_rrd=[]
V_rrd=[]
F_rrd=[]
Q_rrd=[]
N_rrd_1=[]
S_rrd_1=[]
V_rrd_1=[]
F_rrd_1=[]
Q_rrd_1=[]
N_rrd_2=[]
S_rrd_2=[]
V_rrd_2=[]
F_rrd_2=[]
Q_rrd_2=[]
N_row=[]
S_row=[]
V_row=[]
F_row=[]
Q_row=[]
N_row_1=[]
S_row_1=[]
V_row_1=[]
F_row_1=[]
Q_row_1=[]
N_row_2=[]
S_row_2=[]
V_row_2=[]
F_row_2=[]
Q_row_2=[]
qg=64
num=0
print("load data.........")
tic=time.time()
for i in range(len(s)):
    a =s[i][:-4]
    if a!='102' and a!='104' and a!='107' and a!='110':
        num = num + 1
        print(data_path+'/'+ a)
        record = wfdb.rdrecord(data_path + '/' + a, sampfrom=0, channel_names=['MLII'])
        sigal = record.p_signal
        annotation = wfdb.rdann(data_path+'/' + a,'atr')
        for j in range(annotation.ann_len-1):
            if annotation.sample[j]>=145 and annotation.sample[j]+215<=sigal.shape[0]:
                rrs=(annotation.sample[j+1]-annotation.sample[j-1])/qg
                rr_pre=annotation.sample[j]-annotation.sample[j-1]
                rr_post=annotation.sample[j+1]-annotation.sample[j]
                rrd=math.floor((rr_pre-rr_post)/rrs)
                sign=sigal[annotation.sample[j]-145:annotation.sample[j]+216]
                sign=resample(sign, qg)
                sig = np.zeros(shape=sign.shape[0])
                sigd = np.zeros(shape=sign.shape[0])
                bmax=np.max(sign)
                bmin = np.min(sign)
                qs=(bmax-bmin)/qg
                for l in range(sign.shape[0]):
                    #print(l)
                    if a!='116':
                        if sig[l]!=bmax:
                            sig[l]=math.floor((sign[l]-bmin)/qs)/qg
                        else:
                            sig[l]=1
                    else:
                        if sig[l]!=bmax:
                            sig[l]=(sign[l]-bmin/qs)/qg#无法转换
                        else:
                            sig[l]=1
                for b in range(sign.shape[0]):
                    if b==0:
                        sigd[b]=sig[b]-0
                    else:
                        sigd[b] = sig[b] - sig[b-1]
                if annotation.symbol[j]=='N' or annotation.symbol[j]=='L' or annotation.symbol[j]=='R' or annotation.symbol[j]=='e' or annotation.symbol[j]=='j':
                    N_sig_2.append(sigd)
                    N_rrd_2.append(rrd)
                    N_row_2.append(sig)
                    if annotation.sample[j]<=108000:
                        N_sig.append(sigd)
                        N_rrd.append(rrd)
                        N_row.append(sig)
                    else:
                        N_sig_1.append(sigd)
                        N_rrd_1.append(rrd)
                        N_row_1.append(sig)
                elif annotation.symbol[j] == 'A' or annotation.symbol[j] == 'a' or annotation.symbol[j] == 'J' :
                    S_sig_2.append(sigd)
                    S_rrd_2.append(rrd)
                    S_row_2.append(sign)
                    if annotation.sample[j]<=108000:
                        S_sig.append(sigd)
                        S_rrd.append(rrd)
                        S_row.append(sign)
                    else:
                        S_sig_1.append(sigd)
                        S_rrd_1.append(rrd)
                        S_row_1.append(sign)
                elif annotation.symbol[j] == 'V' or annotation.symbol[j] == 'E':
                    V_sig_2.append(sigd)
                    V_rrd_2.append(rrd)
                    V_row_2.append(sign)
                    if annotation.sample[j]<=108000:
                        V_sig.append(sigd)
                        V_rrd.append(rrd)
                        V_row.append(sign)
                    else:
                        V_sig_1.append(sigd)
                        V_rrd_1.append(rrd)
                        V_row_1.append(sign)
                elif annotation.symbol[j] == 'F' :
                    F_sig_2.append(sigd)
                    F_rrd_2.append(rrd)
                    F_row_2.append(sign)
                    if annotation.sample[j]<=108000:
                        F_sig.append(sigd)
                        F_rrd.append(rrd)
                        F_row.append(sign)
                    else:
                        F_sig_1.append(sigd)
                        F_rrd_1.append(rrd)
                        F_row_1.append(sign)
                elif annotation.symbol[j] == 'Q':
                    Q_sig_2.append(sigd)
                    Q_rrd_2.append(rrd)
                    Q_row_2.append(sign)
                    if annotation.sample[j]<=108000:
                        Q_sig.append(sigd)
                        Q_rrd.append(rrd)
                        Q_row.append(sign)
                    else:
                        Q_sig_1.append(sigd)
                        Q_rrd_1.append(rrd)
                        Q_row_1.append(sign)
            else:
                continue
    else:
        continue
    if num==20:
        print('save 20 feature data.....',a)
        N_sig_2 = np.asarray(N_sig_2, dtype=np.float32)
        S_sig_2 = np.asarray(S_sig_2, dtype=np.float32)
        V_sig_2 = np.asarray(V_sig_2, dtype=np.float32)
        F_sig_2 = np.asarray(F_sig_2, dtype=np.float32)
        Q_sig_2 = np.asarray(Q_sig_2, dtype=np.float32)
        np.save('D:/python/rnn/npy_64/N_seg20.npy', N_sig_2)
        np.save('D:/python/rnn/npy_64/S_seg20.npy', S_sig_2)
        np.save('D:/python/rnn/npy_64/V_seg20.npy', V_sig_2)
        np.save('D:/python/rnn/npy_64/F_seg20.npy', F_sig_2)
        np.save('D:/python/rnn/npy_64/Q_seg20.npy', Q_sig_2)
        N_sig_2 = []
        S_sig_2 = []
        V_sig_2 = []
        F_sig_2 = []
        Q_sig_2 = []
        N_rrd_2 = np.asarray(N_rrd_2, dtype=np.float32)
        S_rrd_2 = np.asarray(S_rrd_2, dtype=np.float32)
        V_rrd_2 = np.asarray(V_rrd_2, dtype=np.float32)
        F_rrd_2 = np.asarray(F_rrd_2, dtype=np.float32)
        Q_rrd_2 = np.asarray(Q_rrd_2, dtype=np.float32)
        np.save('D:/python/rnn/npy_64/N_rrd20.npy', N_rrd_2)
        np.save('D:/python/rnn/npy_64/S_rrd20.npy', S_rrd_2)
        np.save('D:/python/rnn/npy_64/V_rrd20.npy', V_rrd_2)
        np.save('D:/python/rnn/npy_64/F_rrd20.npy', F_rrd_2)
        np.save('D:/python/rnn/npy_64/Q_rrd20.npy', Q_rrd_2)
        N_rrd_2 = []
        S_rrd_2 = []
        V_rrd_2 = []
        F_rrd_2 = []
        Q_rrd_2 = []
        N_row_2 = np.asarray(N_row_2, dtype=np.float32)
        S_row_2 = np.asarray(S_row_2, dtype=np.float32)
        V_row_2 = np.asarray(V_row_2, dtype=np.float32)
        F_row_2 = np.asarray(F_row_2, dtype=np.float32)
        Q_row_2 = np.asarray(Q_row_2, dtype=np.float32)
        np.save('D:/python/rnn/npy_64/N_row20.npy', N_row_2)
        np.save('D:/python/rnn/npy_64/S_row20.npy', S_row_2)
        np.save('D:/python/rnn/npy_64/V_row20.npy', V_row_2)
        np.save('D:/python/rnn/npy_64/F_row20.npy', F_row_2)
        np.save('D:/python/rnn/npy_64/Q_row20.npy', Q_row_2)
        N_row_2 = []
        S_row_2 = []
        V_row_2 = []
        F_row_2 = []
        Q_row_2 = []
N_sig = np.asarray(N_sig, dtype=np.float32)
S_sig = np.asarray(S_sig, dtype=np.float32)
V_sig = np.asarray(V_sig, dtype=np.float32)
F_sig = np.asarray(F_sig, dtype=np.float32)
Q_sig = np.asarray(Q_sig, dtype=np.float32)
np.save('D:/python/rnn/npy_64/N_5min_seg.npy', N_sig)
np.save('D:/python/rnn/npy_64/S_5min_seg.npy', S_sig)
np.save('D:/python/rnn/npy_64/V_5min_seg.npy', V_sig)
np.save('D:/python/rnn/npy_64/F_5min_seg.npy', F_sig)
np.save('D:/python/rnn/npy_64/Q_5min_seg.npy', Q_sig)
N_row = np.asarray(N_row, dtype=np.float32)
S_row = np.asarray(S_row, dtype=np.float32)
V_row = np.asarray(V_row, dtype=np.float32)
F_row = np.asarray(F_row, dtype=np.float32)
Q_row = np.asarray(Q_row, dtype=np.float32)
np.save('D:/python/rnn/npy_64/N_5min_row.npy', N_row)
np.save('D:/python/rnn/npy_64/S_5min_row.npy', S_row)
np.save('D:/python/rnn/npy_64/V_5min_row.npy', V_row)
np.save('D:/python/rnn/npy_64/F_5min_row.npy', F_row)
np.save('D:/python/rnn/npy_64/Q_5min_row.npy', Q_row)
N_row_1 = np.asarray(N_row_1, dtype=np.float32)
S_row_1 = np.asarray(S_row_1, dtype=np.float32)
V_row_1 = np.asarray(V_row_1, dtype=np.float32)
F_row_1 = np.asarray(F_row_1, dtype=np.float32)
Q_row_1 = np.asarray(Q_row_1, dtype=np.float32)
np.save('D:/python/rnn/npy_64/N_25min_row.npy', N_row_1)
np.save('D:/python/rnn/npy_64/S_25min_row.npy', S_row_1)
np.save('D:/python/rnn/npy_64/V_25min_row.npy', V_row_1)
np.save('D:/python/rnn/npy_64/F_25min_row.npy', F_row_1)
np.save('D:/python/rnn/npy_64/Q_25min_row.npy', Q_row_1)
N_sig_1 = np.asarray(N_sig_1, dtype=np.float32)
S_sig_1 = np.asarray(S_sig_1, dtype=np.float32)
V_sig_1 = np.asarray(V_sig_1, dtype=np.float32)
F_sig_1 = np.asarray(F_sig_1, dtype=np.float32)
Q_sig_1 = np.asarray(Q_sig_1, dtype=np.float32)
np.save('D:/python/rnn/npy_64/N_25min_seg.npy', N_sig_1)
np.save('D:/python/rnn/npy_64/S_25min_seg.npy', S_sig_1)
np.save('D:/python/rnn/npy_64/V_25min_seg.npy', V_sig_1)
np.save('D:/python/rnn/npy_64/F_25min_seg.npy', F_sig_1)
np.save('D:/python/rnn/npy_64/Q_25min_seg.npy', Q_sig_1)
N_rrd = np.asarray(N_rrd, dtype=np.float32)
S_rrd= np.asarray(S_rrd, dtype=np.float32)
V_rrd = np.asarray(V_rrd, dtype=np.float32)
F_rrd = np.asarray(F_rrd, dtype=np.float32)
Q_rrd = np.asarray(Q_rrd, dtype=np.float32)
print(N_sig_1.shape)
print(S_sig_1.shape)
print(V_sig_1.shape)
print(F_sig_1.shape)
print(Q_sig_1.shape)
print(N_row_1.shape)
print(S_row_1.shape)
print(V_row_1.shape)
print(F_row_1.shape)
print(Q_row_1.shape)
print(np.mean(N_rrd))
print(np.mean(S_rrd))
print(np.mean(V_rrd))
print(np.mean(F_rrd))
print(np.mean(Q_rrd))
np.save('D:/python/rnn/npy_64/N_5min_rrd.npy',N_rrd)
np.save('D:/python/rnn/npy_64/S_5min_rrd.npy',S_rrd)
np.save('D:/python/rnn/npy_64/V_5min_rrd.npy',V_rrd)
np.save('D:/python/rnn/npy_64/F_5min_rrd.npy',F_rrd)
np.save('D:/python/rnn/npy_64/Q_5min_rrd.npy',Q_rrd)
N_rrd_1 = np.asarray(N_rrd_1, dtype=np.float32)
S_rrd_1 = np.asarray(S_rrd_1, dtype=np.float32)
V_rrd_1 = np.asarray(V_rrd_1, dtype=np.float32)
F_rrd_1 = np.asarray(F_rrd_1, dtype=np.float32)
Q_rrd_1 = np.asarray(Q_rrd_1, dtype=np.float32)
np.save('D:/python/rnn/npy_64/N_25min_rrd.npy', N_rrd_1)
np.save('D:/python/rnn/npy_64/S_25min_rrd.npy', S_rrd_1)
np.save('D:/python/rnn/npy_64/V_25min_rrd.npy', V_rrd_1)
np.save('D:/python/rnn/npy_64/F_25min_rrd.npy', F_rrd_1)
np.save('D:/python/rnn/npy_64/Q_25min_rrd.npy', Q_rrd_1)
draw_ecg(N_sig[0,:])