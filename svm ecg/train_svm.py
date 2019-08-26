from feature import *
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from ecg import *
from evaluation_AAMI import *
from keras.utils import to_categorical
use_probability = False
multi_mode = 'ovo'
target_class = ['train', 'test']
model_class = ['RR', 'HOS','lbp','Morph','wav','lstm']
model_svm_path='D:/python/svm/model_t/'
c_value=[0.001,0.01,0.1,1,10,100]
judge=0
sig_class= ['N', 'S','V','F']
for i in range(len(target_class)):
    if os.path.isfile('D:/python/svm/npy_180/data_' + target_class[i] + '_seg.npy') \
            and os.path.isfile('D:/python/svm/npy_180/label_' + target_class[i] + '_seg.npy') \
            and os.path.isfile('D:/python/svm/npy_180/R_' + target_class[i] + '.npy')\
            and os.path.isfile('D:/python/svm/npy_180/valid_R_' + target_class[i] + '.npy'):
        judge=judge+1
if judge!=2:
    print('creat row ecg data...')
    creat_data()
else:
    print('loading row ecg data...')

def creat_label():
    class_ID= np.load('D:/python/svm/npy_180/label_'+target_class[0]+'_seg.npy')
    labels_tr = np.array(sum(class_ID, [])).flatten()
    class_ID= np.load('D:/python/svm/npy_180/label_'+target_class[1]+'_seg.npy')
    labels_te = np.array(sum(class_ID, [])).flatten()
    print("loading labels...")
    return labels_tr,labels_te

def creat_feature(features_labels_name):
    if os.path.isfile('D:/python/svm/feature/f_'+features_labels_name+'_train'+'.npy') \
            and os.path.isfile('D:/python/svm/feature/f_'+features_labels_name+'_test'+'.npy'):
        print("Loading feature: " + 'f_' + features_labels_name + "...")
        f_tr=np.load('D:/python/svm/feature/f_'+features_labels_name+'_train.npy')
        f_te = np.load('D:/python/svm/feature/f_' + features_labels_name + '_test.npy')
    elif features_labels_name=='RR':
        f_tr,f_te=rr_feature()
    elif features_labels_name == 'HOS':
        f_tr, f_te = HOS()
    elif features_labels_name=='lbp':
        f_tr,f_te=LBP()
    elif features_labels_name=='Morph':
        f_tr,f_te=Morph()
    elif features_labels_name=='wav':
        f_tr,f_te=wavelets()
    elif features_labels_name=='lstm':
        f_tr,f_te=lstm_feature()
    return f_tr,f_te


for i in range(len(model_class)):
    i=5
    l_tr, l_te = creat_label()
    f_tr,f_te=creat_feature(model_class[i])
    if os.path.isfile('D:/python/svm/feature/f_tr_' + model_class[i] + '_smote' + '.npy') \
            and os.path.isfile('D:/python/svm/feature/l_tr_' + model_class[i] + '_smote' + '.npy'):
        print('loading '+model_class[i]+' SMOTE data...')
        f_tr=np.load('D:/python/svm/feature/f_tr_' + model_class[i] + '_smote' + '.npy')
        l_tr=np.load('D:/python/svm/feature/l_tr_' + model_class[i] + '_smote' + '.npy')
    else:
        f_tr,l_tr=perform_oversampling('SMOTE',f_tr,l_tr,model_class[i])
        np.save('D:/python/svm/feature/f_tr_' + model_class[i] + '_smote' + '.npy',f_tr)
        np.save('D:/python/svm/feature/l_tr_' + model_class[i] + '_smote' + '.npy', l_tr)
    scaler = StandardScaler()
    scaler.fit(f_tr)
    f_tr_scaled = scaler.transform(f_tr)
    f_te_scaled = scaler.transform(f_te)
    for k in range(len(c_value)):
        if os.path.isfile(model_svm_path+model_class[i]+'_SVM_'+str(c_value[k])+'.pkl'):
            print('loading SVM model...')
            svm_model = joblib.load(model_svm_path+model_class[i]+'_SVM_'+str(c_value[k])+'.pkl')
        else:
            print(model_class[i]+' SVM_'+str(c_value[k])+' training....')
            start = time.time()
            class_weights = {}
            for c in range(4):
                class_weights.update({c: len(l_tr) / float(np.count_nonzero(l_tr == c))})
            svm_model = svm.SVC(C=c_value[k], kernel='rbf', degree=3, gamma='auto',
                                coef0=0.0, shrinking=True, probability=use_probability, tol=0.001,
                                cache_size=200, class_weight=class_weights, verbose=False,
                                max_iter=-1, decision_function_shape=multi_mode, random_state=None)
            svm_model.fit(f_tr_scaled, l_tr)
            joblib.dump(svm_model, model_svm_path+model_class[i]+'_SVM_'+str(c_value[k])+'.pkl')
            end = time.time()
            print(model_class[i]+'_SVM_'+str(c_value[k])+" trained completed!Time required: " + str(format(end - start, '.2f')) + " sec")
        print('predicting and saving train data....')
        start = time.time()
        pred_vt = svm_model.predict(f_tr_scaled)
        pred_vt=to_categorical(pred_vt,num_classes=len(sig_class))
        l_tr_test = to_categorical(l_tr, num_classes=len(sig_class))
        roc_probs = np.ndarray.sum(pred_vt, axis=1)
        pred_v = np.argmax(pred_vt, axis=1)
        true_v = np.argmax(l_tr_test, axis=1)
        plot_confusion_matrix(true_v, pred_v, np.array(sig_class))
        print_results(true_v, pred_v, sig_class,'D:/python/svm/result/'+model_class[i]+'_result_SVM_'+str(c_value[k])+'_train.txt')
        plt.savefig('D:/python/svm/confusion_matrix/'+model_class[i]+'_result_SVM_'+str(c_value[k])+'_train.png')
        plt.show()
        end = time.time()
        print("Time required: " + str(format(end - start, '.2f')) + " sec")
        print('predicting and saving test data....')
        start = time.time()
        pred_vt = svm_model.predict(f_te_scaled)
        pred_vt=to_categorical(pred_vt,num_classes=len(sig_class))
        l_te_test = to_categorical(l_te, num_classes=len(sig_class))
        roc_probs = np.ndarray.sum(pred_vt, axis=1)
        pred_v = np.argmax(pred_vt, axis=1)
        true_v = np.argmax(l_te_test, axis=1)
        plot_confusion_matrix(true_v, pred_v, np.array(sig_class))
        print_results(true_v, pred_v, sig_class,'D:/python/svm/result/'+model_class[i]+'_result_SVM_'+str(c_value[k])+'_test.txt')
        plt.savefig('D:/python/svm/confusion_matrix/'+model_class[i]+'_result_SVM_'+str(c_value[k])+'_test.png')
        plt.show()
        end = time.time()
        print("Time required: " + str(format(end - start, '.2f')) + " sec")