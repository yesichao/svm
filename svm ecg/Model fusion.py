from feature import *
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from ecg import *
from keras.utils import to_categorical
from evaluation_AAMI import *
model_class = ['RR', 'HOS','lbp','Morph','wav']
model_svm_path='D:/python/svm/model_t/'
c_value=[0.001,0.01,0.1,1,10,100]
sig_class= ['N', 'S','V','F']
output_path='D:/python/svm/row predict/'
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
    return f_tr,f_te
def creat_test_data(i):
    l_tr, l_te = creat_label()
    f_tr, f_te = creat_feature(model_class[i])
    if os.path.isfile('D:/python/svm/feature/f_tr_' + model_class[i] + '_smote' + '.npy') \
            and os.path.isfile('D:/python/svm/feature/l_tr_' + model_class[i] + '_smote' + '.npy'):
        print('loading ' + model_class[i] + ' SMOTE data...')
        f_tr = np.load('D:/python/svm/feature/f_tr_' + model_class[i] + '_smote' + '.npy')
        l_tr = np.load('D:/python/svm/feature/l_tr_' + model_class[i] + '_smote' + '.npy')
    else:
        f_tr, l_tr = perform_oversampling('SMOTE', f_tr, l_tr, model_class[i])
        np.save('D:/python/svm/feature/f_tr_' + model_class[i] + '_smote' + '.npy', f_tr)
        np.save('D:/python/svm/feature/l_tr_' + model_class[i] + '_smote' + '.npy', l_tr)
    scaler = StandardScaler()
    scaler.fit(f_tr)
    f_tr_scaled = scaler.transform(f_tr)
    f_te_scaled = scaler.transform(f_te)
    print('success for loading labels....')
    return f_te_scaled,l_te
def creat_label():
    class_ID= np.load('D:/python/svm/npy_180/label_'+target_class[0]+'_seg.npy')
    labels_tr = np.array(sum(class_ID, [])).flatten()
    class_ID= np.load('D:/python/svm/npy_180/label_'+target_class[1]+'_seg.npy')
    labels_te = np.array(sum(class_ID, [])).flatten()
    print("loading labels...")
    return labels_tr,labels_te
#0-RR,1-HOS,2-lbp,3-Morph,4-wav
if os.path.isfile(output_path + 'prob_ovo_RR.csv') and \
        os.path.isfile(output_path + 'prob_ovo_HOS.csv') and \
        os.path.isfile(output_path + 'prob_ovo_wvl.csv') and \
        os.path.isfile(output_path + 'prob_ovo_Morph.csv') and \
        os.path.isfile(output_path + 'prob_ovo_LBP.csv'):
    print("loading row predict data....")
    start = time.time()
    prob_ovo_RR         = np.loadtxt(output_path + 'prob_ovo_RR.csv')
    prob_ovo_wvl        = np.loadtxt(output_path + 'prob_ovo_wvl.csv')
    prob_ovo_LBP        = np.loadtxt(output_path + 'prob_ovo_LBP.csv')
    prob_ovo_HOS        = np.loadtxt(output_path + 'prob_ovo_HOS.csv')
    prob_ovo_Morph    = np.loadtxt(output_path + 'prob_ovo_Morph.csv')
    end = time.time()
    print("Time required: " + str(format(end - start, '.2f')) + " sec")
else:
    print('creating row predict data...')
    start = time.time()
    RR_svm_model = joblib.load(model_svm_path + 'RR_SVM_0.001' + '.pkl')
    wav_svm_model = joblib.load(model_svm_path + 'wav_SVM_0.001' + '.pkl')
    HOS_svm_model = joblib.load(model_svm_path + 'HOS_SVM_0.001' + '.pkl')
    Morph_svm_model = joblib.load(model_svm_path + 'Morph_SVM_0.001' + '.pkl')
    lbp_svm_model = joblib.load(model_svm_path + 'lbp_SVM_0.001' + '.pkl')
    RR_feature,RR_l_te=creat_test_data(0)
    prob_ovo_RR = RR_svm_model.decision_function(RR_feature)
    wav_feature,wav_l_te=creat_test_data(4)
    prob_ovo_wvl= wav_svm_model.decision_function(wav_feature)
    HOS_feature,HOS_l_te=creat_test_data(1)
    prob_ovo_HOS = HOS_svm_model.decision_function(HOS_feature)
    Morph_feature,Morph_l_te=creat_test_data(3)
    prob_ovo_Morph = Morph_svm_model.decision_function(Morph_feature)
    lbp_feature,lbp_l_te=creat_test_data(2)
    prob_ovo_LBP = lbp_svm_model.decision_function(lbp_feature)
    np.savetxt(output_path + 'prob_ovo_RR.csv', prob_ovo_RR)
    np.savetxt(output_path + 'prob_ovo_wvl.csv', prob_ovo_wvl)
    np.savetxt(output_path + 'prob_ovo_HOS.csv', prob_ovo_HOS)
    np.savetxt(output_path + 'prob_ovo_Morph.csv', prob_ovo_Morph)
    np.savetxt(output_path + 'prob_ovo_LBP.csv', prob_ovo_LBP)
    end = time.time()
    print("Time required: " + str(format(end - start, '.2f')) + " sec")
print('Calculation fusion result...')
start = time.time()
predict, prob_ovo_RR_sig = ovo_voting_exp(prob_ovo_RR,4)
predict, prob_ovo_wvl_sig = ovo_voting_exp(prob_ovo_wvl,4)
predict, prob_ovo_HOS_sig = ovo_voting_exp(prob_ovo_HOS,4)
predict, prob_ovo_MyDescp_sig = ovo_voting_exp(prob_ovo_Morph, 4)
predict, prob_ovo_LBP_sig  = ovo_voting_exp(prob_ovo_LBP, 4)
probs_ensemble = np.stack((prob_ovo_RR_sig, prob_ovo_wvl_sig, prob_ovo_HOS_sig, prob_ovo_MyDescp_sig))
end = time.time()
print("Time required: " + str(format(end - start, '.2f')) + " sec")
print('Drawing ...')
predictions_prob_rule = basic_rules(probs_ensemble, 0)
pred_vt = to_categorical(predictions_prob_rule.astype(int), num_classes=len(sig_class))
l_tr_test,l_te_test=creat_label()
l_te_test = to_categorical(l_te_test, num_classes=len(sig_class))
roc_probs = np.ndarray.sum(pred_vt, axis=1)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(l_te_test, axis=1)
plot_confusion_matrix(true_v, pred_v, np.array(sig_class))
print_results(true_v, pred_v, sig_class,
              'D:/python/svm/result/fusion_result_SVM.txt')
plt.savefig('D:/python/svm/confusion_matrix/fusion_result_SVM.png')
plt.show()
