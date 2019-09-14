from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import time
from net_1 import *
from utils import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
data_path = 'D:/python/rnn/npy_64/'

# --------------------- 数据载入和整理 -------------------------------
train_data = {'N': 'N_seg20.npy',
              'S': 'S_seg20.npy',
              'V': 'V_seg20.npy',
              'F': 'F_seg20.npy',
              'Q': 'Q_seg20.npy'}
train_data_1 = {'N': 'N_5min_seg.npy',
              'S': 'S_5min_seg.npy',
              'V': 'V_5min_seg.npy',
              'F': 'F_5min_seg.npy',
              'Q': 'Q_5min_seg.npy'}
test_data = {'N': 'N_25min_seg.npy',
             'S': 'S_25min_seg.npy',
             'V': 'V_25min_seg.npy',
             'F': 'F_25min_seg.npy',
             'Q': 'Q_25min_seg.npy'}
train_rr_data = {'N': 'N_rrd20.npy',
              'S': 'S_rrd20.npy',
              'V': 'V_rrd20.npy',
              'F': 'F_rrd20.npy',
              'Q': 'Q_rrd20.npy'}
train_rr_data_1 = {'N': 'N_5min_rrd.npy',
              'S': 'S_5min_rrd.npy',
              'V': 'V_5min_rrd.npy',
              'F': 'F_5min_rrd.npy',
              'Q': 'Q_5min_rrd.npy'}
test_rr_data = {'N': 'N_25min_rrd.npy',
             'S': 'S_25min_rrd.npy',
             'V': 'V_25min_rrd.npy',
             'F': 'F_25min_rrd.npy',
             'Q': 'Q_25min_rrd.npy'}
train_row_data = {'N': 'N_row20.npy',
              'S': 'S_row20.npy',
              'V': 'V_row20.npy',
              'F': 'F_row20.npy',
              'Q': 'Q_row20.npy'}
train_row_data_1 = {'N': 'N_5min_row.npy',
              'S': 'S_5min_row.npy',
              'V': 'V_5min_row.npy',
              'F': 'F_5min_row.npy',
              'Q': 'Q_5min_row.npy'}
test_row_data = {'N': 'N_25min_row.npy',
             'S': 'S_25min_row.npy',
             'V': 'V_25min_row.npy',
             'F': 'F_25min_row.npy',
             'Q': 'Q_25min_row.npy'}
target_class = ['N', 'S','V', 'F', 'Q']
target_num=[75,75,75,13,7]
tic = time.time()
for i in range(len(target_class)):
    TrainXt = np.load(data_path + train_data[target_class[i]])
    Trainrr = np.load(data_path + train_rr_data[target_class[i]])
    Trainrr = Trainrr.reshape(-1, 1)
    Indices = np.arange(TrainXt.shape[0])  # 随机打乱索引
    np.random.shuffle(Indices)
    TrainXt=TrainXt[Indices]
    TrainXt = TrainXt[0:target_num[i],:]
    Trainrr=Trainrr[Indices]
    Trainrr=Trainrr[0:target_num[i],:]
    #Trainrow = Trainrow.reshape(-1, 64)
    TrainX = np.load(data_path + train_data_1[target_class[i]])
    TrainXr = np.load(data_path + train_rr_data_1[target_class[i]])
    TrainX = TrainX.reshape(-1,64)
    TrainXr=TrainXr.reshape(-1,1)
    TrainX=np.concatenate((TrainX,TrainXt))
    TrainXr= np.concatenate((TrainXr, Trainrr))
    TestXt = np.load(data_path + test_data[target_class[i]])
    TestXr = np.load(data_path + test_rr_data[target_class[i]])
    TestXr = TestXr.reshape(-1, 1)
    #TestXrow = TestXrow.reshape(-1, 64)
    TrainYt = np.array([i]*TrainX.shape[0])
    TestYt = np.array([i]*TestXt.shape[0])
    if i==0:
        x_train=TrainX
        y_train=TrainYt
        rr_train=TrainXr
        rr_test=TestXr
        x_test=TestXt
        y_test=TestYt
    else:
        x_train = np.concatenate((x_train, TrainX))
        rr_train = np.concatenate((rr_train, TrainXr))
        x_test = np.concatenate((x_test, TestXt))
        rr_test=np.concatenate((rr_test, TestXr))
        y_train= np.concatenate((y_train, TrainYt))
        y_test = np.concatenate((y_test, TestYt))
Indices = np.arange(x_train.shape[0])  # 随机打乱索引
np.random.shuffle(Indices)
x_train=x_train[Indices]
y_train=y_train[Indices]
rr_train=rr_train[Indices]
#row_train=row_train[Indices]
x_train = np.expand_dims(x_train, axis=2)
rr_train = np.expand_dims(rr_train, axis=2)
px_train=x_train[:,0:20,:]
rx_train=x_train[:,20:40,:]
tx_train=x_train[:,40:64,:]
Indices = np.arange(x_test.shape[0])  # 随机打乱索引
np.random.shuffle(Indices)
x_test=x_test[Indices]
y_test=y_test[Indices]
rr_test=rr_test[Indices]
x_test = np.expand_dims(x_test, axis=2)
rr_test = np.expand_dims(rr_test, axis=2)
px_test=x_test[:,0:20,:]
rx_test=x_test[:,20:40,:]
tx_test=x_test[:,40:64,:]
print(x_train.shape)
print(x_test.shape)
y_train = to_categorical(y_train, num_classes=len(target_class))
y_test = to_categorical(y_test, num_classes=len(target_class))

toc = time.time()
print('Time for data processing--- '+str(toc-tic)+' seconds---')
# ------------------------ 网络生成与训练 ----------------------------
model = Net()

model_name = 'D:/python/rnn/model_t/myNet_6.h5'
checkpoint = ModelCheckpoint(filepath=model_name,
                             monitor='val_categorical_accuracy', mode='max',
                             save_best_only='True')
callback_lists = [checkpoint]
model.fit(x={'main_inputs': x_train, 'p_inputs':px_train,'r_inputs':rx_train,'t_inputs':tx_train,'rr_inputs':rr_train},
            y=y_train, batch_size=16, epochs=100,
          verbose=1,validation_data=({'main_inputs': x_test, 'p_inputs':px_test,'r_inputs':rx_test,'t_inputs':tx_test,'rr_inputs':rr_test}, y_test), callbacks=callback_lists)
model.save( 'D:/python/rnn/model_t/myNet_1.h5')
pred_vt = model.predict({'main_inputs': x_train, 'p_inputs':px_train,'r_inputs':rx_train,'t_inputs':tx_train,'rr_inputs':rr_train}, batch_size=16, verbose=1)
print(pred_vt)
roc_probs = np.ndarray.sum(pred_vt, axis=1)
print(roc_probs)
print(roc_probs.shape)
np.save('D:/python/rnn/'+'pred'+ '_' + 'test' + '.npy', pred_vt)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(y_train, axis=1)

plot_confusion_matrix(true_v, pred_v, np.array(target_class))
print_results(true_v, pred_v, target_class)
plt.show()
