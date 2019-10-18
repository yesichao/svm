from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import time
from net import *
from utils import *
import time
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from net import *
import matplotlib.pyplot as plt
from keras.callbacks import Callback,ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score
target_class=['N','S','V','F','Q']
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
data_path = 'D:/python/liangji/row_data/'
data_path_1 = 'D:/python/liangji/feature/'

class Metrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        pred_vt = model.predict(
            {'main_inputs': x_test, 'A4_inputs': A4_test, 'D2_inputs': D2_test, 'D3_inputs': D3_test,
             'D4_inputs': D4_test, 'mix_inputs': mix_test}, batch_size=128, verbose=1)
        print(pred_vt)
        roc_probs = np.ndarray.sum(pred_vt, axis=1)
        print(roc_probs)
        print(roc_probs.shape)
        np.save('D:/python/liangji/' + 'pred' + '_' + 'test' + '.npy', pred_vt)
        pred_v = np.argmax(pred_vt, axis=1)
        true_v = np.argmax(y_test, axis=1)

        plot_confusion_matrix(true_v, pred_v, np.array(target_class))
        print_results(true_v, pred_v, target_class,'sen_p+_1.txt')
        plt.show()
        return
metrics = Metrics()
# --------------------- 数据载入和整理 -------------------------------
sig=np.load('D:/python/liangji/row_data/sig_train_sig.npy')
sig_1=np.load('D:/python/liangji/row_data/sig_5min_sig.npy')
l_tr=np.load('D:/python/liangji/row_data/label_train_sig.npy')
l_tr_1=np.load('D:/python/liangji/row_data/label_5min_sig.npy')
A4=np.load('D:/python/liangji/feature/A4_train.npy')
A4_1=np.load('D:/python/liangji/feature/A4_5min.npy')
A4_train=np.concatenate((A4,A4_1))
D2=np.load('D:/python/liangji/feature/D2_train.npy')
D2_1=np.load('D:/python/liangji/feature/D2_5min.npy')
D2_train=np.concatenate((D2,D2_1))
D3=np.load('D:/python/liangji/feature/D3_train.npy')
D3_1=np.load('D:/python/liangji/feature/D3_5min.npy')
D3_train=np.concatenate((D3,D3_1))
D4=np.load('D:/python/liangji/feature/D4_train.npy')
D4_1=np.load('D:/python/liangji/feature/D4_5min.npy')
D4_train=np.concatenate((D4,D4_1))
mix=np.load('D:/python/liangji/feature/mix_train.npy')
mix_1=np.load('D:/python/liangji/feature/mix_5min.npy')
RR=np.load('D:/python/liangji/row_data/RR_train_sig.npy')
RR_1=np.load('D:/python/liangji/row_data/RR_5min_sig.npy')
mix_train=np.concatenate((np.concatenate((mix,RR),axis=1),np.concatenate((mix_1,RR_1),axis=1)))
A4_train = np.expand_dims(A4_train, axis=2)
D2_train = np.expand_dims(D2_train, axis=2)
D3_train = np.expand_dims(D3_train, axis=2)
D4_train = np.expand_dims(D4_train, axis=2)
mix_train = np.expand_dims(mix_train, axis=2)
x_train=np.concatenate((sig,sig_1))
y_train=np.concatenate((l_tr,l_tr_1)).reshape(-1,1)
y_train=to_categorical(y_train, num_classes=len(target_class))
Indices = np.arange(x_train.shape[0])  # 随机打乱索引
np.random.shuffle(Indices)
x_train=x_train[Indices]
y_train=y_train[Indices]
A4_train=A4_train[Indices]
D2_train=D2_train[Indices]
D3_train=D3_train[Indices]
D4_train=D4_train[Indices]
mix_train=mix_train[Indices]
print(x_train.shape,A4_train.shape,D2_train.shape,D3_train.shape,D4_train.shape,mix_train.shape)
print(y_train.shape)
x_test=np.load('D:/python/liangji/row_data/sig_test_sig.npy')
y_test=np.load('D:/python/liangji/row_data/label_test_sig.npy').reshape(-1,1)
y_test=to_categorical(y_test, num_classes=len(target_class))
A4_test=np.load('D:/python/liangji/feature/A4_test.npy')
D2_test=np.load('D:/python/liangji/feature/D2_test.npy')
D3_test=np.load('D:/python/liangji/feature/D3_test.npy')
D4_test=np.load('D:/python/liangji/feature/D4_test.npy')
mix_t=np.load('D:/python/liangji/feature/mix_test.npy')
RR_t=np.load('D:/python/liangji/row_data/RR_test_sig.npy')
mix_test=np.concatenate((mix_t,RR_t),axis=1)
A4_test = np.expand_dims(A4_test, axis=2)
D2_test = np.expand_dims(D2_test, axis=2)
D3_test = np.expand_dims(D3_test, axis=2)
D4_test = np.expand_dims(D4_test, axis=2)
mix_test = np.expand_dims(mix_test, axis=2)
print(x_test.shape,A4_test.shape,D2_test.shape,D3_test.shape,D4_test.shape,mix_test.shape)
print(y_test.shape)
# ------------------------ 网络生成与训练 ----------------------------
model = Net()

model_name = 'D:/python/liangji/model_t/myNet_1.h5'
checkpoint = ModelCheckpoint(filepath=model_name,
                             monitor='val_categorical_accuracy', mode='max',
                             save_best_only='True')
callback_lists = [checkpoint,metrics]
model.fit(x={'main_inputs': x_train, 'A4_inputs':A4_train,'D2_inputs':D2_train,'D3_inputs':D3_train,'D4_inputs':D4_train,'mix_inputs':mix_train},
            y=y_train, batch_size=128, epochs=100,
          verbose=1,validation_data=({'main_inputs': x_test, 'A4_inputs':A4_test,'D2_inputs':D2_test,'D3_inputs':D3_test,'D4_inputs':D4_test,'mix_inputs':mix_test}, y_test), callbacks=callback_lists)
model.save( 'D:/python/rnn/model_t/myNet_1.h5')
