from features_ECG import *
from keras.models import load_model
from lstm_train import *
model_name = 'D:/python/svm/model_t/myNet_1.h5'
if os.path.isfile(model_name):
    print('loading lstm model...')
    base_model = load_model(model_name)
else:
    print('training lstm model...')
    base_model = lstm_model()
for i in range(len(target_class)):
    sig = np.load('D:/python/svm/npy_180/sig_' + target_class[i] + '_seg.npy')
    class_ID= np.load('D:/python/svm/npy_180/label_'+target_class[i]+'_seg.npy')
    labels_tr = np.array(sum(class_ID, [])).flatten()
    labels_tr = to_categorical(labels_tr, num_classes=len(sig_class))
    sig = np.expand_dims(sig, axis=2)
    pred_vt = base_model.predict(sig, batch_size=16, verbose=1)
    print(pred_vt)
    roc_probs = np.ndarray.sum(pred_vt, axis=1)
    print(roc_probs)
    print(roc_probs.shape)
    pred_v = np.argmax(pred_vt, axis=1)
    true_v = np.argmax(labels_tr, axis=1)
    plot_confusion_matrix(true_v, pred_v, np.array(sig_class))
    print_results(true_v, pred_v, sig_class,'D:/python/svm/result/'+target_class[i]+'_result_SVM_test.txt')
    plt.show()