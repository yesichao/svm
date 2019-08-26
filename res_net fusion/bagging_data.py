import time
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from net_2 import *
import matplotlib.pyplot as plt
from ecg_seg import *
from utils import *
from keras.callbacks import Callback,ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
F1=[]
Se=[]
Pp=[]
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict( self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average=None)
        _val_recall = recall_score(val_targ, val_predict,average=None)

        _val_precision = precision_score(val_targ, val_predict,average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        F1.append(_val_f1)
        Se.append(_val_recall)
        Pp.append(_val_precision)
        print("-val_f1:",_val_f1)
        print("-val_recall:", _val_recall)
        print("-val_mean_recall:", np.mean(_val_recall))
        print("-val_precision:", _val_precision)
        return
metrics = Metrics()

data_path = 'D:/python/bwl res_net/npy_170/'
# --------------------- 数据载入和整理 -------------------------------

train_data = {'N': 'N_train_seg.npy',
              'S': 'S_train_seg.npy',
              'V': 'V_train_seg.npy',
              'F': 'F_train_seg.npy'}

test_data = {'N': 'N_test_seg.npy',
             'S': 'S_test_seg.npy',
             'V': 'V_test_seg.npy',
             'F': 'F_test_seg.npy'}
target_class = ['N', 'S', 'V','F']

print('processing data....')
tic=time.time()
#creat_data()
length=1440
for i in range(len(target_class)):
    TrainXt = np.load(data_path + train_data[target_class[i]])
    print(TrainXt.shape)
    TrainXt=TrainXt.reshape(-1,length)
    Indices = np.arange(TrainXt.shape[0])  # 随机打乱索引
    np.random.shuffle(Indices)
    TrainXt = TrainXt[Indices]
    x=int(TrainXt.shape[0]/3)
    for j in range(3):
        data=TrainXt[x*j:x*(j+1),:]
        np.save('D:/python/bwl res_net/bagging_data/'+train_data[target_class[i]][:-4]+"_"+str(j)+".npy",data)
