#!/usr/bin/env python
# Rui Zhang 8.2020
# rui.zhang@cern.ch

def get_name():
    return 'example'

def get_model(**args):
    """Example model from keras documentation"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))
    return model
