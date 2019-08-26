from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from net import *
from utils  import *
from l import *
DS1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
data = {'train': DS1,
        'test': DS2
        }
size_RR_max=20
winL=90
winR=90
target_class = ['train', 'test']
sig_class = ['N', 'S', 'V', 'F']
target_class = ['train', 'test']
def lstm_model():
    if os.path.isfile('D:/python/svm/npy_180/sig_' + target_class[0] + '_seg.npy') and \
            os.path.isfile('D:/python/svm/npy_180/sig_' + target_class[1] + '_seg.npy'):
        sig_train=np.load('D:/python/svm/npy_180/sig_' + target_class[0] + '_seg.npy')
        sig_test=np.load('D:/python/svm/npy_180/sig_' + target_class[1] + '_seg.npy')
    else:
        sig_train,sig_test=creat_sig()
    class_ID = np.load('D:/python/svm/npy_180/label_' + target_class[0] + '_seg.npy')
    labels_tr = np.array(sum(class_ID, [])).flatten()
    class_ID = np.load('D:/python/svm/npy_180/label_' + target_class[1] + '_seg.npy')
    labels_te = np.array(sum(class_ID, [])).flatten()
    Indices = np.arange(sig_train.shape[0])  # 随机打乱索引
    np.random.shuffle(Indices)
    sig_train=sig_train[Indices]
    labels_tr=labels_tr[Indices]
    sig_train= np.expand_dims(sig_train, axis=2)
    sig_test = np.expand_dims(sig_test, axis=2)
    labels_tr = to_categorical(labels_tr, num_classes=len(sig_class))
    labels_te = to_categorical(labels_te, num_classes=len(sig_class))
    model = Net()
    model_name = 'D:/python/svm/model_t/myNet_1.h5'
    checkpoint = ModelCheckpoint(filepath=model_name,
                                 monitor='val_categorical_accuracy', mode='max',
                                 save_best_only='True')
    callback_lists = [checkpoint]
    model.fit(x=sig_train, y=labels_tr, batch_size=16, epochs=50,
              verbose=1, validation_data=(sig_test, labels_te), callbacks=callback_lists)
    return model