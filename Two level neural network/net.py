from keras.layers import AveragePooling1D,Flatten
from keras.models import Model
from keras.models import Input
from keras.layers import Conv1D,concatenate
from keras.layers.core import Dense
from keras.optimizers import SGD,Adam
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
    lr = 0.001*((1-0.0001)**epoch)
    print('Learning rate: ', lr)
    return lr


def Net():
    main_inputs =  Input(shape=[180,1],
                   dtype='float32',
                   name='main_inputs')
    layer1=Conv1D(32, 9, strides=1, padding='same',activation='relu')(main_inputs)
    layer2=MaxPooling1D(pool_size=2, strides=4)(layer1)
    layer3=Conv1D(16, 9, strides=1, padding='same',activation='relu')(layer2)
    layer4 = Dropout(0.2)(layer3)
    main_output =Flatten()(layer4)
    A4_inputs =  Input(shape=[44,1],
                   dtype='float32',
                   name='A4_inputs')
    A4_layer=Conv1D(4, 8, strides=1, padding='same',activation='relu')(A4_inputs)
    A4_layer = AveragePooling1D(pool_size=2, strides=2)(A4_layer)
    A4_output =Flatten()(A4_layer)
    D2_inputs =  Input(shape=[44,1],
                   dtype='float32',
                   name='D2_inputs')
    D2_layer=Conv1D(4, 8, strides=1, padding='same',activation='relu')(D2_inputs)
    D2_layer = AveragePooling1D(pool_size=2, strides=2)(D2_layer)
    D2_output =Flatten()(D2_layer)
    D3_inputs =  Input(shape=[44,1],
                   dtype='float32',
                   name='D3_inputs')
    D3_layer=Conv1D(4, 8, strides=1, padding='same',activation='relu')(D3_inputs)
    D3_layer = AveragePooling1D(pool_size=2, strides=2)(D3_layer)
    D3_output =Flatten()(D3_layer)
    D4_inputs =  Input(shape=[44,1],
                   dtype='float32',
                   name='D4_inputs')
    D4_layer=Conv1D(4, 8, strides=1, padding='same',activation='relu')(D4_inputs)
    D4_layer = AveragePooling1D(pool_size=2, strides=2)(D4_layer)
    D4_output =Flatten()(D4_layer)
    mix_inputs= Input(shape=[9,1],
                   dtype='float32',
                   name='mix_inputs')
    mix_output = Flatten()(mix_inputs)
    x=concatenate([main_output,A4_output,D2_output,D3_output,D4_output,mix_output],axis=-1)
    layer = Dense(20)(x)
    output=Dense(5,activation='softmax')(layer)
    model = Model(inputs=[main_inputs,A4_inputs,D2_inputs,D3_inputs,D4_inputs,mix_inputs], outputs=[output])
    optimizer =Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    model.summary()
    return model