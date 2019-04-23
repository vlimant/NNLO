import mpiLAPI
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
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
    def __init__(self, input_shape=(150,94,5)):
        self.input_shape = input_shape
        return

    def build(self, params):
        kernel_size = params[0]
        dropout = params[1]
        lr = 10.0 ** params[2]

        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, kernel_size=(kernel_size, kernel_size), strides=3, activation='relu')(inputs)
        x = Conv2D(32, kernel_size=(kernel_size, kernel_size), strides=3, activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout / 2)(x)
        x = Flatten(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=Adam(lr=lr),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model.to_json()

    def get_parameter_grid(self):
        return [
            (3, 9),   # kernel_size
            (.0, .5), # dropout
            (-5, 1),  # lr
        ]
    
    def get_name(self):
       return "CNN_model"

class DenseNetModel(BaseModel):
    def __init__(self, input_shape=(150,94,5)):
        self.input_shape = input_shape
        return
    
    def build(self, params):
        depth = params[0]
        nb_dense_block = params[1]
        growth_rate = params[2]
        dropout_rate = params[3]
        nb_filter = params[4]
        lr = 10. ** params[5]
        densenet = DenseNet(nb_classes=3, img_dim=self.input_shape, depth=depth, nb_dense_block=nb_dense_block, 
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


if __name__ == '__main__':
    arch = CNNModel()
    m = arch.build([0, 0, 0])
    print(m)
