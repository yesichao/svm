import warnings
import wfdb
import time
from utils import *
import math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
target_sig_length=3600
warnings.filterwarnings("ignore")
data_path='D:/python/ecg/MIT-BIH'
s=['200','202','210','213','214','219','221','228','231','233','234','212','222','232']#２００、２０２、２１０、２１３、２１４、２１９、２２１、２２８、２３１、２３３和２３４
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
    a =s[i]
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
                    N_row_2.append(sign)
                    if annotation.sample[j]<=108000:
                        N_sig.append(sigd)
                        N_rrd.append(rrd)
                        N_row.append(sign)
                    else:
                        N_sig_1.append(sigd)
                        N_rrd_1.append(rrd)
                        N_row_1.append(sign)
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
                elif annotation.symbol[j] == 'V':
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

N_row_1 = np.asarray(N_row_1, dtype=np.float32)
S_row_1 = np.asarray(S_row_1, dtype=np.float32)
V_row_1 = np.asarray(V_row_1, dtype=np.float32)
F_row_1 = np.asarray(F_row_1, dtype=np.float32)
Q_row_1 = np.asarray(Q_row_1, dtype=np.float32)
np.save('D:/python/rnn/npy_64_test/N_25min_row.npy', N_row_1)
np.save('D:/python/rnn/npy_64_test/S_25min_row.npy', S_row_1)
np.save('D:/python/rnn/npy_64_test/V_25min_row.npy', V_row_1)
np.save('D:/python/rnn/npy_64_test/F_25min_row.npy', F_row_1)
np.save('D:/python/rnn/npy_64_test/Q_25min_row.npy', Q_row_1)
N_sig_1 = np.asarray(N_sig_1, dtype=np.float32)
S_sig_1 = np.asarray(S_sig_1, dtype=np.float32)
V_sig_1 = np.asarray(V_sig_1, dtype=np.float32)
F_sig_1 = np.asarray(F_sig_1, dtype=np.float32)
Q_sig_1 = np.asarray(Q_sig_1, dtype=np.float32)
np.save('D:/python/rnn/npy_64_test/N_25min_seg.npy', N_sig_1)
np.save('D:/python/rnn/npy_64_test/S_25min_seg.npy', S_sig_1)
np.save('D:/python/rnn/npy_64_test/V_25min_seg.npy', V_sig_1)
np.save('D:/python/rnn/npy_64_test/F_25min_seg.npy', F_sig_1)
np.save('D:/python/rnn/npy_64_test/Q_25min_seg.npy', Q_sig_1)
N_rrd_1 = np.asarray(N_rrd_1, dtype=np.float32)
S_rrd_1 = np.asarray(S_rrd_1, dtype=np.float32)
V_rrd_1 = np.asarray(V_rrd_1, dtype=np.float32)
F_rrd_1 = np.asarray(F_rrd_1, dtype=np.float32)
Q_rrd_1 = np.asarray(Q_rrd_1, dtype=np.float32)
np.save('D:/python/rnn/npy_64_test/N_25min_rrd.npy', N_rrd_1)
np.save('D:/python/rnn/npy_64_test/S_25min_rrd.npy', S_rrd_1)
np.save('D:/python/rnn/npy_64_test/V_25min_rrd.npy', V_rrd_1)
np.save('D:/python/rnn/npy_64_test/F_25min_rrd.npy', F_rrd_1)
np.save('D:/python/rnn/npy_64_test/Q_25min_rrd.npy', Q_rrd_1)
