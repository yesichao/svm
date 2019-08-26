from keras.layers import Add
from keras.layers import MaxPooling1D
from keras.layers.core import Lambda
from keras.models import Model
from keras import regularizers
from keras.models import Input
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.layers import Dropout,Flatten
from keras.layers.core import Dense, Activation
from keras.layers import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.layers import Reshape, CuDNNLSTM, Bidirectional
def lr_schedule(epoch):
    lr = 0.1
    if epoch >= 20 and epoch < 40:
        lr = 0.01
    if epoch >= 40:
        lr = 0.001
    print('Learning rate: ', lr)
    return lr

def Net():
    inputs =  Input(shape=[180,1],
                   dtype='float32',
                   name='inputs')
    layer1=Conv1D(14, 24, strides=1, padding='same',activation='relu')(inputs)
    layer2=MaxPooling1D(pool_size=2, strides=4)(layer1)
    layer3=Bidirectional(LSTM(32, return_sequences=True), merge_mode='concat')(layer2)
    layer4 = Dropout(0.2)(layer3)
    layer5 = Flatten()(layer4)
    layer6 = Dense(20)(layer5)
    output = Dense(4,activation='sigmoid')(layer6)
    model = Model(inputs=[inputs], outputs=[output])
    optimizer = SGD(lr=lr_schedule(0), momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    model.summary()
    return model
