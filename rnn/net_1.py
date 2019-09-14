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
from keras.layers import Dropout
from keras.layers.core import Dense, Activation
from keras.layers import GlobalAveragePooling1D
from keras.optimizers import SGD

def lr_schedule(epoch):
    lr = 0.001*((1-0.0001)**epoch)
    print('Learning rate: ', lr)
    return lr

conv_subsample_lengths = [1, 2, 1, 2, 1, 2, 1, 2]
def Net():
    main_inputs =  Input(shape=[64,1],
                   dtype='float32',
                   name='main_inputs')
    conv1=Conv1D(12,32,strides=1, padding='same', kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.001))(main_inputs)
    layer1=BatchNormalization()(conv1)
    layer=Activation('tanh')(layer1)
    for index, subsample_length in enumerate(conv_subsample_lengths):
        def zeropad(x):
            y = K.zeros_like(x)
            return K.concatenate([x, y], axis=2)

        def zeropad_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 3
            shape[2] *= 2
            return tuple(shape)

        num_filters = 2**int(index / 2) \
                     * 12
        shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
        zero_pad = (index % 2) == 0 \
            and index > 0
        if zero_pad is True:
            shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
        for i in range(2):
             if not (index == 0 and i == 0):
                 layer = BatchNormalization()(layer)
                 layer = Activation("relu")(layer)
                 if i > 0:
                     layer = Dropout(0.5)(layer)
                 else:
                     pass
             s=subsample_length if i == 0 else 1
             layer = Conv1D(num_filters, 32, strides=s, padding='same',
                                    kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(layer)
        layer = Add()([shortcut, layer])
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = AveragePooling1D()(layer)
    main_output =Flatten()(layer)
    p_inputs =  Input(shape=[20,1],
                   dtype='float32',
                   name='p_inputs')
    p_layer=Conv1D(4, 3, strides=1, padding='same',activation='relu')(p_inputs)
    p_layer = AveragePooling1D(pool_size=2, strides=2)(p_layer)
    p_output =Flatten()(p_layer)
    r_inputs =  Input(shape=[20,1],
                   dtype='float32',
                   name='r_inputs')
    r_layer=Conv1D(4, 3, strides=1, padding='same',activation='relu')(r_inputs)
    r_layer = AveragePooling1D(pool_size=2, strides=2)(r_layer)
    r_output =Flatten()(r_layer)
    t_inputs =  Input(shape=[24,1],
                   dtype='float32',
                   name='t_inputs')
    t_layer=Conv1D(4, 3, strides=1, padding='same',activation='relu')(t_inputs)
    t_layer = AveragePooling1D(pool_size=2, strides=2)(t_layer)
    t_output =Flatten()(t_layer)
    rr_inputs= Input(shape=[1,1],
                   dtype='float32',
                   name='rr_inputs')
    rr_output = Flatten()(rr_inputs)
    x=concatenate([main_output,p_output,r_output,t_output,rr_output],axis=-1)
    layer = Dense(10)(x)
    output=Dense(5,activation='softmax')(layer)
    model = Model(inputs=[main_inputs, p_inputs,r_inputs,t_inputs,rr_inputs], outputs=[output])
    optimizer =Adam(lr=lr_schedule(0))
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    model.summary()
    return model
