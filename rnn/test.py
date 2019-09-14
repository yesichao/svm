from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import time
from net import *
from utils import *
from keras.models import load_model
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
data_path = 'D:/python/rnn/npy_64_test/'

# --------------------- 数据载入和整理 -------------------------------
test_data = {'N': 'N_25min_seg.npy',
             'S': 'S_25min_seg.npy',
             'V': 'V_25min_seg.npy',
             'F': 'F_25min_seg.npy',
             'Q': 'Q_25min_seg.npy'}
test_rr_data = {'N': 'N_25min_rrd.npy',
             'S': 'S_25min_rrd.npy',
             'V': 'V_25min_rrd.npy',
             'F': 'F_25min_rrd.npy',
             'Q': 'Q_25min_rrd.npy'}
test_row_data = {'N': 'N_25min_row.npy',
             'S': 'S_25min_row.npy',
             'V': 'V_25min_row.npy',
             'F': 'F_25min_row.npy',
             'Q': 'Q_25min_row.npy'}
target_class = ['N', 'S','V', 'F', 'Q']
target_num=[75,75,75,13,7]
tic = time.time()
for i in range(len(target_class)):
    TestXt = np.load(data_path + test_data[target_class[i]])
    TestXr = np.load(data_path + test_rr_data[target_class[i]])
    TestXr = TestXr.reshape(-1, 1)
    #TestXrow = TestXrow.reshape(-1, 64)
    TestYt = np.array([i]*TestXt.shape[0])
    if i==0:
        rr_test=TestXr
        x_test=TestXt
        y_test=TestYt
    else:
        x_test = np.concatenate((x_test, TestXt))
        rr_test=np.concatenate((rr_test, TestXr))
        y_test = np.concatenate((y_test, TestYt))
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
print(x_test.shape)
y_test = to_categorical(y_test, num_classes=len(target_class))

toc = time.time()
print('Time for data processing--- '+str(toc-tic)+' seconds---')
# ------------------------ 网络生成与训练 ----------------------------
model = load_model( 'D:/python/rnn/model_t/myNet_6.h5')
pred_vt = model.predict({'main_inputs': x_test, 'p_inputs':px_test,'r_inputs':rx_test,'t_inputs':tx_test,'rr_inputs':rr_test}, batch_size=16, verbose=1)
print(pred_vt)
roc_probs = np.ndarray.sum(pred_vt, axis=1)
print(roc_probs)
print(roc_probs.shape)
np.save('D:/python/rnn/'+'pred'+ '_' + 'test' + '.npy', pred_vt)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(y_test, axis=1)

plot_confusion_matrix(true_v, pred_v, np.array(target_class))
print_results(true_v, pred_v, target_class)
plt.show()
