from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.noise import GaussianDropout
from keras.regularizers import l2
from keras import backend as K
import os

K.set_image_dim_ordering('th')

from keras.models import model_from_json

def get_model_config(model_name):
    model_config = {}
    if model_name == 'baseline':
        model_config['model_builder'] = baseline_model
    elif model_name == 'simple_CNN':
        model_config['model_builder'] = simple_CNN_model
    elif model_name == 'larger_CNN':
        model_config['model_builder'] = larger_CNN_model
    else:
        raise Exception('Unknown model name.')
    model_config['filepath_weight'] = os.path.join('../data/models', '{}_weight'.format(model_name))
    model_config['filepath_architechture'] = os.path.join('../data/models', '{}_model'.format(model_name))
    return model_config

def baseline_model(num_classes, image_shape):
    model = Sequential()
    model.add(Reshape(int(image_shape[0] * image_shape[1]), input_shape = image_shape))
    model.add(Dense(128, input_dim=128, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    return model

def simple_CNN_model(num_classes, image_shape):
    model = Sequential()
    model.add(Reshape((1, image_shape[0], image_shape[1]), input_shape = image_shape))
    model.add(Convolution2D(32, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def larger_CNN_model(num_classes, image_shape):
    model = Sequential()
    model.add(Reshape((1, image_shape[0], image_shape[1]), input_shape = image_shape))
    model.add(Convolution2D(50, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(40, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def save_model(model, model_config):
    #Saves model weights
    model.save_weights(model_config['filepath_weight'])
    print('Model weights saved in {}.'.format(model_config['filepath_weight']))

    #saves model architechture
    with open(model_config['filepath_architechture'], 'w') as outfile:
        outfile.write(model.to_json())
    print('Model architechture saved in {}.'.format(model_config['filepath_architechture']))

def load_model(filepath_weights, filepath_architechture):
    with open(filepath_architechture, 'r') as read:
        a = read.readlines()
        model = model_from_json(a[0])

    model.load_weights(filepath_weights, by_name=False)

    return model

def get_model_name():
    model_name = input('Select model:(baseline/simple_CNN/[larger_CNN])\n')
    if model_name == '':
        model_name = 'larger_CNN'
    return model_name
