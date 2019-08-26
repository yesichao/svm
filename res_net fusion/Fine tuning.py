import time
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from net import *
import matplotlib.pyplot as plt
from ecg_seg import *
from keras.models import load_model
from utils import *
from Config import Config
from keras.callbacks import Callback,ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = Config()
F1=[]
Se=[]
Pp=[]
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.n=0
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
        if self.n<np.mean(_val_recall):
            self.n=np.mean(_val_recall)
            model.save('D:/python/bwl res_net/model/row_data_se_net.h5')
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
    TestXt = np.load(data_path + test_data[target_class[i]])
    TrainXt=TrainXt.reshape(-1,length)
    TestXt = TestXt.reshape(-1,length)
    TrainYt = np.array([i]*TrainXt.shape[0])
    TestYt = np.array([i]*TestXt.shape[0])
    if i == 0:
        TrainX = multi_prep(TrainXt)
        TestX = multi_prep(TestXt)
        TrainY = TrainYt
        TestY = TestYt
    else:
        TrainX = np.concatenate((TrainX,
                                 multi_prep(TrainXt)))
        TestX = np.concatenate((TestX,
                                multi_prep(TestXt)))
        TrainY = np.concatenate((TrainY, TrainYt))
        TestY = np.concatenate((TestY, TestYt))
Indices = np.arange(TrainX.shape[0])  # 随机打乱索引
np.random.shuffle(Indices)
TrainX = TrainX[Indices]
TrainY = TrainY[Indices]
TrainX = np.expand_dims(TrainX, axis=2)
TestX = np.expand_dims(TestX, axis=2)
print(TestX.shape)
class_weights = {}
for c in range(4):
    class_weights.update({c:TrainY.shape[0]/float(np.count_nonzero(TrainY == c))})

TrainY = to_categorical(TrainY, num_classes=len(target_class))
TestY = to_categorical(TestY, num_classes=len(target_class))
toc = time.time()
print('Time for data processing--- '+str(toc-tic)+' seconds---')

# ------------------------ 网络生成与训练 ----------------------------
#model = Net()
#model_name = 'C:/Users/叶思超/Desktop/结果/1/Data_Enhancement_loss_Net.h5'
def lr_schedule(epoch):
    lr = 0.000001
    if epoch >= 15 and epoch < 40:
        lr = 0.0000001
    if epoch >= 40:
        lr = 0.00000001
    print('Learning rate: ', lr)
    return lr
model_name ='D:/python/bwl res_net/model/row_data_loss_net.h5'
model = load_model(model_name)
model_name = 'D:/python/bwl res_net/fine_model/row_data'
checkpoint = ModelCheckpoint(filepath=model_name+'_loss_Net.h5',
                             monitor='val_loss',
                             save_best_only=True,
                            mode='min')
checkpoint1 = ModelCheckpoint(filepath=model_name+'_acc_Net.h5',
                             monitor='val_categorical_accuracy', mode='max',
                             save_best_only='True')
lr_scheduler = LearningRateScheduler(lr_schedule)
callback_lists = [checkpoint,metrics,checkpoint1,lr_scheduler]

history=model.fit(x=TrainX, y=TrainY, batch_size=128, epochs=50,#class_weight=class_weights,
          verbose=1, validation_data=(TestX, TestY), callbacks=callback_lists)

score = model.evaluate(TestX, TestY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#输出acc_loss曲线
history_dict=history.history
loss_value=history_dict['loss']
val_loss_value=history_dict['val_loss']
print(history_dict.keys())
acc=history_dict['categorical_accuracy']
val_acc=history_dict['val_categorical_accuracy']
epochs=range(1,len(loss_value)+1)
#plt.plot(epochs,loss_value,'r',label='Training loss')
#plt.plot(epochs,val_loss_value,'b',label='Validation loss')
plt.plot(epochs,acc,'g',label='Training acc')
plt.plot(epochs,val_acc,'k',label='Validation acc')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('D:/python/bwl res_net/result/acc_loss_1.png')
plt.show()
history_dict=history.history
loss_value=history_dict['loss']
val_loss_value=history_dict['val_loss']
print(history_dict.keys())
#acc=history_dict['categorical_accuracy']
#val_acc=history_dict['val_categorical_accuracy']
epochs=range(1,len(loss_value)+1)
plt.plot(epochs,loss_value,'r',label='Training loss')
plt.plot(epochs,val_loss_value,'b',label='Validation loss')
#plt.plot(epochs,acc,'g',label='Training acc')
#plt.plot(epochs,val_acc,'k',label='Validation acc')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('D:/python/bwl res_net/result/acc_loss_2.png')
plt.show()
#输出F1,Se,P+曲线
val_recalls = np.asarray(Se, dtype=np.float32)
val_precisions = np.asarray(Pp, dtype=np.float32)
val_f1s = np.asarray(F1, dtype=np.float32)
print(val_f1s.shape)
epochs=range(1,len(loss_value)+1)
plt.plot(epochs,np.mean(val_recalls,axis=1),'r',label='Se')
plt.plot(epochs,np.mean(val_precisions,axis=1),'b',label='P+')
plt.plot(epochs,np.mean(val_f1s,axis=1),'g',label='F1')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('score')
plt.legend()
plt.savefig('D:/python/bwl res_net/result/score_val_1.png')
plt.show()
#5类se变化
epochs=range(1,len(loss_value)+1)
plt.plot(epochs,val_recalls[:,0],'r',label='N')
plt.plot(epochs,val_recalls[:,1],'b',label='S')
plt.plot(epochs,val_recalls[:,2],'g',label='V')
plt.plot(epochs,val_recalls[:,3],'m',label='F')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('Se')
plt.legend()
plt.show()
#5类P+变化
epochs=range(1,len(loss_value)+1)
plt.plot(epochs,val_precisions[:,0],'r',label='N')
plt.plot(epochs,val_precisions[:,1],'b',label='S')
plt.plot(epochs,val_precisions[:,2],'g',label='V')
plt.plot(epochs,val_precisions[:,3],'m',label='F')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('P+')
plt.legend()
plt.show()
#5类F1变化
epochs=range(1,len(loss_value)+1)
plt.plot(epochs,val_f1s[:,0],'r',label='N')
plt.plot(epochs,val_f1s[:,1],'b',label='S')
plt.plot(epochs,val_f1s[:,2],'g',label='V')
plt.plot(epochs,val_f1s[:,3],'m',label='F')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('F1')
plt.legend()
plt.show()
pred_vt = model.predict(TestX, batch_size=64, verbose=1)
print(pred_vt)
roc_probs = np.ndarray.sum(pred_vt, axis=1)
print(roc_probs)
print(roc_probs.shape)
np.save('D:/python/bwl res_net/'+'pred'+ '_' + 'test' + '.npy', pred_vt)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(TestY, axis=1)

plot_confusion_matrix(true_v, pred_v, np.array(target_class))
print_results(true_v, pred_v, target_class,'D:/python/bwl res_net/result/sen_p+_1.txt')
plt.savefig('D:/python/bwl res_net/result/confusion_matrix_1.png')
plt.show()
f = open('D:/python/bwl res_net/result/score_1.txt', "w")

f.write('  Se = ' + str(val_recalls) + "\n")
f.write('  P+ = ' + str(val_precisions) + "\n")
f.write('  F1 = ' + str(val_f1s) + "\n")

f.close()