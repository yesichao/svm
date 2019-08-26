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
x_train=np.load("D:/python/imgs_train.npy")
y_train=np.load("D:/python/imgs_mask_train.npy")
x_val=np.load("D:/python/imgs_val.npy")
y_val=np.load("D:/python/imgs_mask_val.npy")
x_test=np.load("D:/python/imgs_test.npy")
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")
print(np.max(x_train))
print(np.min(x_train))
print("load model and test...")
tic=time.time()
MODEL_PATH = 'D:/python/model_t/'
model_name = 'unet_model_' + str(1) + '.hdf5'
model = load_model(MODEL_PATH + model_name,{'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
pred_vt = model.predict(x_val,batch_size=2, verbose=1)
pred_tt = model.predict(x_test,batch_size=2, verbose=1)
pred_vt[pred_vt > 0.5] = 1
pred_vt[pred_vt <= 0.5] = 0
pred_v=pred_vt
true_v = y_val
pred_tt[pred_tt > 0.5] = 1
pred_tt[pred_tt <= 0.5] = 0
np.save('imgs_mask_val_1.npy', pred_v)
np.save('imgs_mask_test.npy', pred_tt)
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")
