import os
import numpy as np
import net
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import time
from net import *
from utils import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
data_path = 'D:/python/rnn/npy_64/'

# --------------------- 数据载入和整理 -------------------------------
N_sig=np.load('D:/python/rnn/npy_64/N_5min_row.npy')
draw_ecg(N_sig[0,:])