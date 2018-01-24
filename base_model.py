import mpiLAPI
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from densenet import DenseNet


class BaseModel:
    def __init__(self):
        assert NotImplementedError
    
    def build(self, params):
        assert NotImplementedError
    
    def get_parameter_grid(self):
        assert NotImplementedError

    def get_name(self):
        assert NotImplementedError

class CNNModel(BaseModel):
    def __init__(self):
        return

    def build(self, params):
        kernel_size = params[0]
        dropout = params[1]
        lr = 10.0 ** params[2]

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(kernel_size, kernel_size), strides=3, activation='relu', input_shape=(150,94,5)))
        model.add(Conv2D(32, kernel_size=(kernel_size, kernel_size), strides=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout / 2))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr))
        return model.to_json()

    def get_parameter_grid(self):
        return [
            (3, 9),   # kernel_size
            (.0, .5), # dropout
            (-5, 1),  # lr
        ]
    
    def get_name(self):
        return "CNN"

class DenseNetModel(BaseModel):
    def __init__(self):
        return
    
    def build(self, params):
        depth = params[0]
        nb_dense_block = params[1]
        growth_rate = params[2]
        dropout_rate = params[3]
        nb_filter = params[4]
        lr = 10. ** params[5]
        densenet = DenseNet(nb_classes=3, img_dim=(150, 94, 5), depth=depth, nb_dense_block=nb_dense_block, 
                        growth_rate = growth_rate, dropout_rate = dropout_rate, nb_filter = nb_filter)
        optimizer = Adam(lr = lr)
        densenet.compile(loss='categorical_crossentropy', optimizer = optimizer)
        return densenet.to_json()

    def get_parameter_grid(self):
        return [
            (10, 10),  # depth
            (3, 3),    # nb_dense_block
            (12, 12),  # growth_rate
            (.0, .0),  # dropout 
            (16, 16),  # nb_filter
            (-5, 1),   # lr
        ]
        
    def get_name(self):
        return "DenseNet"

