from keras.layers import AveragePooling1D,Flatten
from keras.models import Model
from keras.models import Input
from keras.layers import Conv1D,concatenate
from keras.layers.core import Dense
from keras.optimizers import SGD,Adam

def lr_schedule(epoch):
    lr = 0.001*((1-0.0001)**epoch)
    print('Learning rate: ', lr)
    return lr


def Net():
    main_inputs =  Input(shape=[64,1],
                   dtype='float32',
                   name='main_inputs')
    main_layer=Conv1D(32, 3, strides=1, padding='same',activation='relu')(main_inputs)
    main_layer = AveragePooling1D(pool_size=2, strides=2)(main_layer)
    main_layer=Conv1D(16, 3, strides=1, padding='same',activation='relu')(main_layer)
    main_layer = AveragePooling1D(pool_size=2, strides=2)(main_layer)
    main_output =Flatten()(main_layer)
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