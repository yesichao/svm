import os
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import time
from net import *
from keras.models import load_model
from utils  import *
def lstm_feature():
    model_name='D:/python/svm/model_t/myNet.h5'
    if os.path.isfile(model_name):
        model = load_model(model_name)
