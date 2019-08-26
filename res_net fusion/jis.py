import os
from glob import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import time
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import collections
from sklearn import svm
import numpy as np
from keras import backend as K
import pywt
target_names = ['N', 'S', 'V','F','Q']
cm=[]
a=[]
for i in [39157,931,1284,2816,50]:
    a.append(i)
cm.append(a)
a=[]
for i in [502,1199,252,12,7]:
    a.append(i)
cm.append(a)
a=[]
for i in [284,160,2624,139,13]:
    a.append(i)
cm.append(a)
a=[]
for i in [199,1,110,76,2]:
    a.append(i)
cm.append(a)
a=[]
for i in [2,0,5,0,0]:
    a.append(i)
cm.append(a)

cm = np.asarray(cm, dtype=np.float32)
print(cm)
for i in range(len(target_names)):
    print(target_names[i] + ':')
    Se = cm[i][i] / np.sum(cm[i])
    Pp = cm[i][i] / np.sum(cm[:, i])
    print('  Se = ' + str(Se))
    print('  P+ = ' + str(Pp))
    if i == 0:
        se_mean = Se
        Pp_mean = Pp
    else:
        se_mean = Se + se_mean
        Pp_mean = Pp + Pp_mean
print('  Se_mean = ' + str(se_mean / 4))
print('  P+ = ' + str(Pp_mean / 4))
print('--------------------------------------')