import os
import warnings
import numpy as np
import time
import cv2
from keras.models import Model, load_model
from model import dice_coef_loss,dice_coef
from until import *
epoch=11
test_data_path='D:/US data/test_set/'
test_numb=os.listdir(test_data_path)
model = load_model('D:/python/US/model/model'+str(epoch) + '_net.h5',{'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})

for i in range(len(test_numb)):
    test_data=read_data(test_data_path+test_numb[i])
    pred_vt = model.predict(test_data)
    _, binary = cv2.threshold(pred_vt[0, :, :, 0], 0.5, 1, cv2.THRESH_BINARY)
    '''cv2.imshow(test_numb[i],binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    cv2.imwrite('D:/python/US/result/pre_'+test_numb[i], binary*255)
