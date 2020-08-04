#!/usr/bin/env python
# Rui Zhang 8.2020
# rui.zhang@cern.ch

def get_name():
    return 'cifar10'

def get_model(**args):
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Permute
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D
    import tensorflow.keras.backend as K
    if args:logging.debug("receiving arguments {}".format(args))
    nb_classes = 10
    img_rows, img_cols = 32, 32
    
    # use 1 kernel size for all convolutional layers
    ks = args.get('kernel_size', 3)
    
    # tune the number of filters for each convolution layer
    nb_filters1 = args.get('nb_filters1', 48)
    nb_filters2 = args.get('nb_filters2', 96)
    nb_filters3 = args.get('nb_filters3', 192)
    
    # tune the pool size once
    ps = args.get('pool_size', 2)
    pool_size = (ps,ps)
    
    # tune the dropout rates independently
    do4 = args.get('dropout1', 0.25)
    do5 = args.get('dropout2', 0.5)
    
    # tune the dense layers independently
    dense1 = args.get('dense1', 512)
    dense2 = args.get('dense2', 256)
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)

    #act = 'sigmoid'
    act = 'relu'
        
    i = Input( input_shape)
    l = Conv2D(nb_filters1,( ks, ks), padding='same', activation = act)(i)
    l = MaxPooling2D(pool_size=pool_size)(l)
    #l = Dropout(do1)(l)

    l = Conv2D(nb_filters2, (ks, ks), padding='same',activation=act)(l)
    #l = Conv2D(nb_filters2, (ks, ks))(l)
    l = MaxPooling2D(pool_size=pool_size)(l)
    #l = Dropout(do2)(l)

    l = Conv2D(nb_filters3, (ks, ks), padding='same',activation=act)(l)
    #l = Conv2D(nb_filters3, (ks, ks))(l)
    l = MaxPooling2D(pool_size=pool_size)(l)
    #l = Dropout(do3)(l)

    l = Flatten()(l)
    l = Dense(dense1,activation=act)(l)
    l = Dropout(do4)(l)
    l = Dense(dense2,activation=act)(l)
    l =Dropout(do5)(l)
    
    o = Dense(nb_classes, activation='softmax')(l)

    model = Model(inputs=i, outputs=o)
    #model.summary()
    
    return model

from skopt.space import Real, Integer, Categorical
get_model.parameter_range = [
    Integer(10,300, name='nb_filters1'),
    Integer(10,300, name='nb_filters2'),
    Integer(10,300, name='nb_filters3'),
    Integer(50,1000, name='dense1'),
    Integer(50,1000, name='dense2'),
    Real(0.0, 1.0, name='dropout1'),
    Real(0.0, 1.0, name='dropout2')
]
