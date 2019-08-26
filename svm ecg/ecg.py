import warnings
import wfdb
from features_ECG import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
data_path='D:/python/ecg/MIT-BIH'
DS1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
data = {'train': DS1,
        'test': DS2
        }
size_RR_max=20
target_class = ['train', 'test']
MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']
def creat_data():
    for i in range(len(target_class)):
        s=data[target_class[i]]
        R_poses = [np.array([]) for i in range(len(s))]
        beat = [[] for i in range(len(s))]  # record, beat, lead
        class_ID = [[] for i in range(len(s))]
        valid_R = [np.array([]) for i in range(len(s))]
        for k in range(len(s)):
            print(data_path+'/'+ str(s[k])+target_class[i])
            record = wfdb.rdrecord(data_path + '/' + str(s[k]), sampfrom=0, channel_names=['MLII'])
            sigal = record.p_signal
            record = wfdb.rdrecord(data_path + '/' + str(s[k]), sampfrom=0, channels=[1])
            sigal2=record.p_signal
            sigal=pre_pro(sigal)
            np.save('D:/python/svm/mit-bih/'+ str(s[k])+'.npy',sigal)
            sigal2=pre_pro(sigal2)
            annotation = wfdb.rdann(data_path+'/' + str(s[k]),'atr')
            for j in range(annotation.ann_len-1):
                pos=annotation.sample[j]
                if pos >= size_RR_max and pos + size_RR_max <= sigal.shape[0]:
                    index, value = max(enumerate(sigal[pos - size_RR_max: pos + size_RR_max]), key=operator.itemgetter(1))
                    pos = (pos - size_RR_max) + index
                R_poses[k] = np.append(R_poses[k], pos)
                if annotation.symbol[j] in MITBIH_classes:
                    if (pos > 90 and pos < (len(sigal) - 90)):
                        sign = sigal[pos - 90:pos + 90]
                        sign2=sigal2[pos - 90:pos + 90]
                        beat[k].append((sign,sign2))
                        valid_R[k] = np.append(valid_R[k], 1)
                        if annotation.symbol[j]=='N' or annotation.symbol[j]=='L' or annotation.symbol[j]=='R' :
                            class_ID[k].append(0)
                        elif annotation.symbol[j] == 'A' or annotation.symbol[j] == 'a' or annotation.symbol[j] == 'J' or annotation.symbol[j]=='e' or annotation.symbol[j]=='j' or annotation.symbol[j]=='S':
                            class_ID[k].append(1)
                        elif annotation.symbol[j] == 'V' or annotation.symbol[j] == 'E':
                            class_ID[k].append(2)
                        elif annotation.symbol[j] == 'F' :
                            class_ID[k].append(3)
                    else:
                        valid_R[k] = np.append(valid_R[k], 0)
                else:
                    valid_R[k] = np.append(valid_R[k], 0)
        np.save('D:/python/svm/npy_180/data_'+target_class[i]+'_seg.npy', beat)
        np.save('D:/python/svm/npy_180/label_'+target_class[i]+'_seg.npy', class_ID)
        np.save('D:/python/svm/npy_180/R_'+target_class[i]+'.npy', R_poses)
        np.save('D:/python/svm/npy_180/valid_R_' + target_class[i] + '.npy', valid_R)
        print(len(beat))