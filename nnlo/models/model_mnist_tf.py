#!/usr/bin/env python
# Rui Zhang 8.2020
# rui.zhang@cern.ch

def get_name():
    return 'mnist'

def get_model(**args):
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Permute
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D
    import tensorflow.keras.backend as K
    """MNIST ConvNet from keras/examples/mnist_cnn.py"""
    #np.random.seed(1337)  # for reproducibility
    if args:logging.debug("receiving arguments {}".format(args))
    nb_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = args.get('nb_filters',32)
    # size of pooling area for max pooling
    ps = args.get('pool_size',2)
    
    # convolution kernel size
    ks = args.get('kernel_size',3)
    do = args.get('dropout', 0.25)
    dense = args.get('dense', 128)

    pool_size = (ps,ps)
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Convolution2D(nb_filters, (ks, ks),
                            padding='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, (ks, ks)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(do))
    model.add(Flatten())
    model.add(Dense(dense))
    model.add(Activation('relu'))
    model.add(Dropout(do))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

from skopt.space import Real, Integer, Categorical
get_model.parameter_range =     [
    Integer(10,50, name='nb_filters'),
    Integer(2,10, name='pool_size'),
    Integer(2,10, name='kernel_size'),
    Integer(50,200, name='dense'),
    Real(0.0, 1.0, name='dropout')
]

