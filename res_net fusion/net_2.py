from keras.layers import Add
from keras.layers import MaxPooling1D
from keras.layers.core import Lambda
from keras.models import Model
from keras import regularizers
from keras.models import Input
from keras.layers import Conv1D,Flatten
from keras.layers import BatchNormalization
from keras import backend as K
from keras.layers import Dropout
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD,Adam
from utils import *
from keras.layers import GlobalAveragePooling1D
conv_subsample_lengths = [1, 2, 1, 2, 1, 2, 1, 2]


def lr_schedule(epoch):
    lr = 0.1
    if epoch >= 20 and epoch < 40:
        lr = 0.01
    if epoch >= 40:
        lr = 0.001
    print('Learning rate: ', lr)
    return lr

def identity_block(X,filters,sizes,lam):
    F1, F2 = filters
    S1, S2 = sizes
    shortcut=X
    layer=BatchNormalization()(X)
    layer=Activation('relu')(layer)
    layer = Conv1D(F1, S1, strides=1, padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(rate=0.5)(layer)
    layer = Conv1D(F2, S2, strides=1, padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01))(layer)
    if lam==1:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
    layer = Add()([shortcut, layer])
    return layer
def identity_block_max(X,filters,sizes,lam):
    F1, F2 = filters
    S1,S2=sizes
    shortcut=X
    layer=BatchNormalization()(X)
    layer=Activation('relu')(layer)
    layer = Conv1D(F1, S1, strides=2, padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(rate=0.5)(layer)
    layer = Conv1D(F2, S2, strides=1, padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(layer)
    if lam==1:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
    shortcut=MaxPooling1D(pool_size=2)(shortcut)
    layer = Add()([shortcut, layer])
    return layer
def zeropad(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=2)

def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape[2] *=2
    return tuple(shape)
def Net():
    inputs =  Input(shape=[1440,1],
                   dtype='float32',
                   name='inputs')
    conv1=Conv1D(12,32,strides=1, padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(inputs)
    shortcut = conv1
    layer=BatchNormalization()(conv1)
    layer1=Activation('relu')(layer)
    layer = Conv1D(12, 32, strides=1, padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(layer1)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(rate=0.5)(layer)
    layer = Conv1D(12, 32, strides=1, padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(layer)
    layer = Add()([shortcut, layer])
    layer=identity_block_max(layer,[12,12],[32,32],0)
    layer = identity_block(layer, [24,24],[32,32],1)
    layer = identity_block_max(layer, [24,24],[32,32],0)
    layer = identity_block(layer, [48,48],[32,32],1)
    layer = identity_block_max(layer, [48,48], [32, 32],0)
    layer = identity_block(layer, [96,96],[32,32],1)
    layer = identity_block_max(layer, [96,96], [32, 32],0)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Dense(4)(layer)
    output=Activation('softmax')(layer)
    model = Model(inputs=[inputs], outputs=[output])
    optimizer = optimizer = SGD(lr=lr_schedule(0), momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    model.summary()
    return model