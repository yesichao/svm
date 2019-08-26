import time
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from net import *
import matplotlib.pyplot as plt
from ecg_seg import *
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
val_loss=[]
train_loss=[]
val_acc=[]
train_acc=[]
data_path = 'D:/python/bwl res_net/row_data/'
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
    TrainYt = np.array([i]*TrainXt.shape[0])
    TestYt = np.array([i]*TestXt.shape[0])
    if i == 0:
        TrainX =TrainXt
        TestX = TestXt
        TrainY = TrainYt
        TestY = TestYt
    else:
        TrainX = np.concatenate((TrainX,TrainXt))
        TestX = np.concatenate((TestX,TestXt))
        TrainY = np.concatenate((TrainY, TrainYt))
        TestY = np.concatenate((TestY, TestYt))
TrainY = to_categorical(TrainY, num_classes=len(target_class))
TestY = to_categorical(TestY, num_classes=len(target_class))
toc = time.time()
print('Time for data processing--- '+str(toc-tic)+' seconds---')

# ------------------------ 网络生成与训练 ----------------------------
model = build_network(config)
model.summary()
epoch=50
batch_size=128
nb_classes = 4
x = 0.02
Indices = np.arange(TrainX.shape[0])  # 随机打乱索引
np.random.shuffle(Indices)
TrainX = TrainX[Indices]
TrainY = TrainY[Indices]
gl=0
val_loss_cw=10
val_acc_cw=0
for i in range(1,epoch+1):
    print('training data epoch %d........'%(i))
    tic = time.time()
    n1 = 0
    for j in range(int(TrainX.shape[0]/batch_size)+1):
        tic = time.time()
        cw = []
        if n1 + batch_size <= TrainY.shape[0]:
            y_true = TrainY[n1:n1 + batch_size,:]
            batch_x=TrainX[n1:n1 + batch_size,:,:]
            batch_y=TrainY[n1:n1 + batch_size,:]
            for s in range(nb_classes):
                num = 0
                for a in range(batch_size):
                    if y_true[a][s] == 1:
                        num = num + 1
                cw_class = 1 - num / batch_size + x
                cw.append(cw_class)
            class_weight = {0: cw[0], 1: cw[1], 2: cw[2], 3: cw[3]}
            n1 = n1 + batch_size
        else:
            y_true = TrainY[n1:,:]
            batch_x=TrainX[n1:,:,:]
            batch_y=TrainY[n1:,:]
            for s in range(nb_classes):
                num = 0
                for a in range(TrainY.shape[0] - n1):
                    if y_true[a][s] == 1:
                        num = num + 1
                cw_class = 1 - num / (TrainY.shape[0] - n1) + x
                cw.append(cw_class)
            class_weight = {0: cw[0], 1: cw[1], 2: cw[2], 3: cw[3]}
            n1 = TrainY.shape[0]
        cw = class_weight
        #print(cw)
        loss = model.train_on_batch(batch_x, batch_y,class_weight=cw)
        toc = time.time()
        print('train on ' +str(n1)+'/' +str(TrainY.shape[0])+'  train loss = %.4f' % (loss[0]) + '  train acc = %.4f' % (loss[1]) +"  time is %f sec." % (toc - tic))
    train_loss.append(loss[0])
    train_acc.append(loss[1])
    val_predict = (np.asarray(model.predict(TestX))).round()
    val_targ = TestY
    _val_f1 = f1_score(val_targ, val_predict, average=None)
    _val_recall = recall_score(val_targ, val_predict, average=None)
    _val_precision = precision_score(val_targ, val_predict, average=None)
    F1.append(_val_f1)
    Se.append(_val_recall)
    Pp.append(_val_precision)
    print("-val_f1:", _val_f1)
    print("-val_recall:", _val_recall)
    print("-val_mean_recall:", np.mean(_val_recall))
    print("-val_precision:", _val_precision)
    if gl< np.mean(_val_recall):
        gl=np.mean(_val_recall)
        model.save('D:/python/bwl res_net/model_batch/class_weight_se_net.h5')
    score = model.evaluate(TestX, TestY, verbose=0)
    val_loss.append(score[0])
    val_acc.append(score[1])
    if val_acc_cw< score[1]:
        val_acc_cw=score[1]
        model.save('D:/python/bwl res_net/model_batch/class_weight_acc_net.h5')
    if val_loss_cw>score[0]:
        val_loss_cw=score[0]
        model.save('D:/python/bwl res_net/model_batch/class_weight_loss_net.h5')
    toc = time.time()
    print("epoch %d finsh %f sec." % (i, toc - tic)+'  Test loss:%4f'%(score[0])+'  Test accuracy:%4f'%(score[1]))
#输出acc_loss曲线
epochs=range(1,epoch+1)
#plt.plot(epochs,loss_value,'r',label='Training loss')
#plt.plot(epochs,val_loss_value,'b',label='Validation loss')
plt.plot(epochs,train_acc,'g',label='Training acc')
plt.plot(epochs,val_acc,'k',label='Validation acc')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('D:/python/bwl res_net/result/acc_loss_1.png')
plt.show()
plt.plot(epochs,train_loss,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
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
epochs=range(1,epoch+1)
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
epochs=range(1,epoch+1)
plt.plot(epochs,val_recalls[:,0],'r',label='N')
plt.plot(epochs,val_recalls[:,1],'b',label='S')
plt.plot(epochs,val_recalls[:,2],'g',label='V')
plt.plot(epochs,val_recalls[:,3],'m',label='F')
plt.plot(epochs,val_recalls[:,4],'y',label='Q')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('Se')
plt.legend()
plt.show()
#5类P+变化
epochs=range(1,epoch+1)
plt.plot(epochs,val_precisions[:,0],'r',label='N')
plt.plot(epochs,val_precisions[:,1],'b',label='S')
plt.plot(epochs,val_precisions[:,2],'g',label='V')
plt.plot(epochs,val_precisions[:,3],'m',label='F')
plt.plot(epochs,val_precisions[:,4],'y',label='Q')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('P+')
plt.legend()
plt.show()
#5类F1变化
plt.plot(epochs,val_f1s[:,0],'r',label='N')
plt.plot(epochs,val_f1s[:,1],'b',label='S')
plt.plot(epochs,val_f1s[:,2],'g',label='V')
plt.plot(epochs,val_f1s[:,3],'m',label='F')
plt.plot(epochs,val_f1s[:,4],'y',label='Q')
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