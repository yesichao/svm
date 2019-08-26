import os
import warnings
import numpy as np
import time
from keras.models import Model, load_model
from mri_model import dice_coef_loss,dice_coef
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
def dec(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection ) / (np.sum(y_true_f) + np.sum(y_pred_f) )

def ppv(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection) / (np.sum(y_pred_f))

def sensitivity(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection) / (np.sum(y_true_f))

print("Loading data and labels...")
tic=time.time()

y_val=np.load("D:/python/imgs_mask_val.npy")
y_val_1=np.load("D:/python/imgs_mask_val_1.npy")
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("Elapsed time is %f sec."%(toc-tic))
print("======================================")
print("sorce:")
true_v=y_val.reshape(-1,880,880)
pred_v=y_val_1.reshape(-1,880,880)
num_test = len(y_val)
mean_1 = 0.0
mean_2 = 0.0
mean_3 = 0.0
for i in range(num_test):
    mean_1 += dec(true_v[i,:,:], pred_v[i,:,:])
    mean_2 += ppv(true_v[i,:,:], pred_v[i,:,:])
    mean_3 += sensitivity(true_v[i,:,:], pred_v[i,:,:])
mean_1 /= num_test
mean_2 /= num_test
mean_3 /= num_test
print("DEC: ", mean_1)
print("PPV: ", mean_2)
print("Sensitivity: ", mean_3)
