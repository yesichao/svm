from features_ECG import *
from keras.models import load_model
from lstm_train import *
DS1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
data = {'train': DS1,
        'test': DS2
        }
size_RR_max=20
winL=90
winR=90
target_class = ['train', 'test']
leads_flag = [1,0]
num_leads = np.sum(leads_flag)
def rr_feature():
    print('RR_intervals.....')
    for i in range(len(target_class)):
        s=data[target_class[i]]
        RR = [RR_intervals() for i in range(len(s))]
        R_pos=np.load('D:/python/svm/npy_180/R_'+target_class[i]+'.npy')
        valid_R=np.load('D:/python/svm/npy_180/valid_R_'+target_class[i]+'.npy')
        for p in range(len(s)):
            print('process on '+ str(s[p])+' '+target_class[i])
            RR[p] = compute_RR_intervals(R_pos[p])
            RR[p].pre_R = RR[p].pre_R[(valid_R[p] == 1)]
            RR[p].post_R = RR[p].post_R[(valid_R[p] == 1)]
            RR[p].local_R = RR[p].local_R[(valid_R[p] == 1)]
            RR[p].global_R = RR[p].global_R[(valid_R[p] == 1)]
        f_RR = np.empty((0, 4))
        for p in range(len(RR)):
            row = np.column_stack((RR[p].pre_R, RR[p].post_R, RR[p].local_R, RR[p].global_R))
            f_RR = np.vstack((f_RR, row))
        f_RR_norm = np.empty((0, 4))
        for p in range(len(RR)):
            avg_pre_R = np.average(RR[p].pre_R)
            avg_post_R = np.average(RR[p].post_R)
            avg_local_R = np.average(RR[p].local_R)
            avg_global_R = np.average(RR[p].global_R)
            row = np.column_stack((RR[p].pre_R / avg_pre_R, RR[p].post_R / avg_post_R, RR[p].local_R / avg_local_R,
                                   RR[p].global_R / avg_global_R))
            f_RR_norm = np.vstack((f_RR_norm, row))
        f_rr=np.column_stack((f_RR,f_RR_norm ))
        np.save('D:/python/svm/feature/f_RR_'+target_class[i]+'.npy',f_rr)
        if i==0:
            f_train_rr=f_rr
    return f_train_rr,f_rr
def LBP():
    print("u-lbp ...")
    for i in range(len(target_class)):
        sn=data[target_class[i]]
        f_lbp = np.empty((0, 59 * num_leads))
        features = np.array([], dtype=float)
        sig = np.load('D:/python/svm/npy_180/data_'+target_class[i]+'_seg.npy')
        for p in range(len(sig)):
            print('process on ' + str(sn[p]) +' '+ target_class[i])
            for beat in sig[p]:
                f_lbp_lead = np.empty([])
                for s in range(2):
                    if leads_flag[s] == 1:
                        if f_lbp_lead.size == 1:
                            f_lbp_lead = compute_Uniform_LBP(beat[s], 8)
                        else:
                            f_lbp_lead = np.hstack((f_lbp_lead, compute_Uniform_LBP(beat[s], 8)))
                f_lbp = np.vstack((f_lbp, f_lbp_lead))
        features = np.concatenate((features, f_lbp)) if features.size else f_lbp
        np.save('D:/python/svm/feature/f_lbp_'+target_class[i]+'.npy',features)
        print(features.shape)
        if i==0:
            f_train_lbp=features
    return f_train_lbp,features
def wavelets():
    print("Wavelets ...")
    for i in range(len(target_class)):
        sn=data[target_class[i]]
        f_wav = np.empty((0,23* num_leads))
        features = np.array([], dtype=float)
        sig = np.load('D:/python/svm/npy_180/data_'+target_class[i]+'_seg.npy')
        for p in range(len(sig)):
            print('process on ' + str(sn[p])+' ' + target_class[i])
            for b in sig[p]:
                f_wav_lead = np.empty([])
                for s in range(2):
                    if leads_flag[s] == 1:
                        if f_wav_lead.size == 1:
                            c=b[s].reshape(180)
                            f_wav_lead =  compute_wavelet_descriptor(c, 'db1', 3)
                        else:
                            f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], 'db1', 3)))
                f_wav = np.vstack((f_wav, f_wav_lead))
                #f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))
        features = np.column_stack((features, f_wav))  if features.size else f_wav
        np.save('D:/python/svm/feature/f_wav_' + target_class[i] + '.npy', features)
        print(features.shape)
        if i==0:
            f_train_wav=features
    return f_train_wav,features
def HOS():
    print("HOS ...")
    n_intervals = 6
    lag = int(round((90 + 90) / n_intervals))
    for i in range(len(target_class)):
        sn=data[target_class[i]]
        f_HOS = np.empty((0, (n_intervals - 1) * 2 * num_leads))
        features = np.array([], dtype=float)
        sig = np.load('D:/python/svm/npy_180/data_'+target_class[i]+'_seg.npy')
        for p in range(len(sig)):
            print('process on ' + str(sn[p])+' ' + target_class[i])
            for b in sig[p]:
                f_HOS_lead = np.empty([])
                for s in range(2):
                    if leads_flag[s] == 1:
                        if f_HOS_lead.size == 1:
                            f_HOS_lead = compute_hos_descriptor(b[s], n_intervals, lag)
                        else:
                            f_HOS_lead = np.hstack((f_HOS_lead, compute_hos_descriptor(b[s], n_intervals, lag)))
                f_HOS = np.vstack((f_HOS, f_HOS_lead))
                # f_HOS = np.vstack((f_HOS, compute_hos_descriptor(b, n_intervals, lag)))
        features = np.column_stack((features, f_HOS)) if features.size else f_HOS
        print(features.shape)
        np.save('D:/python/svm/feature/f_HOS_' + target_class[i] + '.npy', features)
        if i==0:
            f_train_HOS=features
    return f_train_HOS,features
def Morph():
    print("My Descriptor ...")
    for i in range(len(target_class)):
        sn=data[target_class[i]]
        f_myMorhp = np.empty((0, 4 * num_leads))
        features = np.array([], dtype=float)
        sig = np.load('D:/python/svm/npy_180/data_'+target_class[i]+'_seg.npy')
        for p in range(len(sig)):
            print('process on ' + str(sn[p]) +' '+ target_class[i])
            for b in sig[p]:
                f_myMorhp_lead = np.empty([])
                for s in range(2):
                    if leads_flag[s] == 1:
                        if f_myMorhp_lead.size == 1:
                            f_myMorhp_lead = compute_my_own_descriptor(b[s], winL, winR)
                        else:
                            f_myMorhp_lead = np.hstack((f_myMorhp_lead, compute_my_own_descriptor(b[s], winL, winR)))
                f_myMorhp = np.vstack((f_myMorhp, f_myMorhp_lead))
                # f_myMorhp = np.vstack((f_myMorhp, compute_my_own_descriptor(b, winL, winR)))
        features = np.column_stack((features, f_myMorhp)) if features.size else f_myMorhp
        print(features.shape)
        np.save('D:/python/svm/feature/f_Morph_' + target_class[i] + '.npy', features)
        if i==0:
            f_train_Morph=features
    return f_train_Morph,features
def lstm_feature():
    model_name='D:/python/svm/model_t/myNet_1.h5'
    if os.path.isfile(model_name):
        print('loading lstm model...')
        base_model = load_model(model_name)
    else:
        print('training lstm model...')
        base_model=lstm_model()
    model=Model(inputs=base_model.input,outputs=base_model.get_layer('dense_1').output)
    for i in range(len(target_class)):
        sig = np.load('D:/python/svm/npy_180/sig_' + target_class[i] + '_seg.npy')
        sig = np.expand_dims(sig, axis=2)
        features = model.predict(sig, batch_size=16, verbose=1)
        np.save('D:/python/svm/feature/f_lstm_' + target_class[i] + '.npy', features)
        if i==0:
            f_train_lstm=features
    return f_train_lstm, features