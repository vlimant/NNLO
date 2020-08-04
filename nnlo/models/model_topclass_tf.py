#!/usr/bin/env python
# Rui Zhang 8.2020
# rui.zhang@cern.ch

def get_name():
    return 'topclass'

def get_model(**args):
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Permute
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D
    if args:logging.debug("receiving arguments {}".format(args))
    conv_layers=args.get('conv_layers',2)
    dense_layers=args.get('dense_layers',2)
    dropout=args.get('dropout',0.2)
    kernel = args.get('kernel_size',3)
    classes=3
    in_channels=5
    in_ch = in_channels
    ## the trace in the input file is 750, 150, 94, 5
    input = Input( (150,94,in_ch))
    ## convs
    c = input
    for i in range(conv_layers):
        channel_in = in_ch*((i+1)%5)
        channel_out = in_ch*((i+2)%5)
        if channel_in == 0: channel_in += 1
        if channel_out == 0: channel_out += 1
        c = Conv2D( filters=channel_out, kernel_size=(kernel,kernel) , strides=1, padding="same", activation = 'relu') (c)
    c = Conv2D(1, (kernel,kernel), activation = 'relu',strides=2, padding="same")(c)

    ## pooling
    pool = args.get('pool', 10)
    m  = MaxPooling2D((pool,pool))(c)
    f = Flatten()(m)
    d = f
    base = args.get('hidden_factor',5)*100
    for i in range(dense_layers):
        N = int(base//(2**(i+1)))
        d = Dense( N, activation='relu')(d)
        if dropout:
            d = Dropout(dropout)(d)
    o = Dense(classes, activation='softmax')(d)

    model = Model(inputs=input, outputs=o)
    #model.summary()
    return model

from skopt.space import Real, Integer, Categorical
get_model.parameter_range = [
    Integer(1,6, name='conv_layers'),
    Integer(1,6, name='dense_layers'),
    Integer(1,6, name='kernel_size'),
    Real(0.0, 1.0, name='dropout')
]
