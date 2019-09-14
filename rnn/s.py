from keras.layers import Add
from keras.layers import MaxPooling1D
from keras.layers.core import Lambda
from keras.models import Model
from keras import regularizers
from keras.models import Input
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.layers import Dropout
from keras.layers.core import Dense, Activation
from keras.layers import GlobalAveragePooling1D
from keras.optimizers import SGD

conv_subsample_lengths = [1, 2, 1, 2, 1, 2, 1, 2]

def lr_schedule(epoch):
    lr = 0.1
    if epoch >= 20 and epoch < 40:
        lr = 0.01
    if epoch >= 40:
        lr = 0.001
    print('Learning rate: ', lr)
    return lr


def Net():
    inputs = Input(shape=[64, 1],
                   dtype='float32',
                   name='inputs')
    layer = BatchNormalization()(inputs)
    conv1 = Conv1D(16, 8, strides=1, padding='same')(layer)
    conv1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.25)(conv1)
    conv2 = Conv1D(32, 4, strides=1, padding='same')(conv1)
    conv2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.25)(conv2)
    conv3 = Conv1D(32, 4, strides=1, padding='same')(conv2)
    conv3 = MaxPooling1D(pool_size=2, strides=2, padding='same')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.25)(conv3)
    conv5 = Conv1D(32, 4, strides=1, padding='same')(conv3)
    conv5 = MaxPooling1D(pool_size=2, strides=2, padding='same')(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.25)(conv5)
    conv6 = Conv1D(32, 4, strides=1, padding='same')(conv5)
    conv6 = MaxPooling1D(pool_size=2, strides=2, padding='same')(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.25)(conv6)
    output=Dense(5, activation='softmax')(conv6)
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model
