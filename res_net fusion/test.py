from keras.models import load_model
from keras.utils import to_categorical
from ecg_seg import *
from utils import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
    class_weights.update({c: TrainY.shape[0]/ float(np.count_nonzero(TrainY == c))})
TrainY = to_categorical(TrainY, num_classes=len(target_class))
TestY = to_categorical(TestY, num_classes=len(target_class))
toc = time.time()
print('Time for data processing--- '+str(toc-tic)+' seconds---')

#model = load_model(MODEL_PATH + model_name,{'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
#model_name = 'C:/Users/叶思超/Desktop/结果/5/class_weight_loss_Net.h5'
model_name ='D:/python/bwl res_net/model/Data_Enhancement_acc_net.h5'
model = load_model(model_name)
model.summary()
pred_vt = model.predict(TestX, batch_size=128, verbose=1)
del TestX
print(pred_vt)
roc_probs = np.ndarray.sum(pred_vt, axis=1)
print(roc_probs)
print(roc_probs.shape)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(TestY, axis=1)

plot_confusion_matrix(true_v, pred_v, np.array(target_class))
print_results(true_v, pred_v, target_class,'D:/python/bwl res_net/result/sen_p+_2.txt')
plt.show()
