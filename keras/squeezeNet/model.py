# SqueezeNet Model Source Code
# Dev. Dongwon Paek

# SqueezeNet Model Source Code

import h5py
import os, shutil
import matplotlib.pyplot as plt
from keras import backend
from keras.models import Model
from keras.layers import Add, Activation, Concatenate, Conv2D, Dropout 
from keras.layers import Flatten, Input, GlobalAveragePooling2D, MaxPooling2D

# This is specific 1.1 version of SqueezeNet. (2.4x less computation according to paper)
# Stacking Layers of SqueezeNet
# nb stands for number
# input_shape   : must have (width, height, dimension)
# nb_classes    : total number of final categories
# dropout_rate  : determines dropout rate after last fire_module. default is None
# compression   : reduce the number of feature maps. Default is 1.0
# RETURENS Keras model instance(keras.models.Model())
def SqueezeNet(input_shape, nb_classes, dropout_rate=None, compression=1.0):
    input_img = Input(shape=input_shape)

    x = Conv2D(int(64*compression), (3, 3), activation='relu', strides=(2, 2), padding='same', name='conv1')(input_img)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1')(x)
    
    x = fire_module(x, int(16 * compression), name='fire2')
    x = fire_module(x, int(16 * compression), name='fire3')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3')(x)
    
    x = fire_module(x, int(32 * compression), name='fire4')
    x = fire_module(x, int(32 * compression), name='fire5')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5')(x)
    
    x = fire_module(x, int(48 * compression), name='fire6')
    x = fire_module(x, int(48 * compression), name='fire7')
    x = fire_module(x, int(64 * compression), name='fire8')
    x = fire_module(x, int(64 * compression), name='fire9')

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    # Creating last conv10
    x = output(x, nb_classes)

    return Model(inputs=input_img, outputs=x)

# Create fire module for SqueezeNet
# x                 : input (keras.layers)
# nb_squeeze_filter : number of filters for Squeezing. Filter size of expanding is 4x of Squeezing filter size
# name              : name of module
# RETURNS fire module x
def fire_module(x, nb_squeeze_filter, name, use_bypass=False):
    nb_expand_filter = 4 * nb_squeeze_filter
    squeeze    = Conv2D(nb_squeeze_filter,(1, 1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
    expand_1x1 = Conv2D(nb_expand_filter, (1, 1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
    expand_3x3 = Conv2D(nb_expand_filter, (3, 3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
    
    if backend.image_data_format() == 'channels_last':
        axis = -1
    else:
        axis = 1

    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
    
    if use_bypass:
        x_ret = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
        
    return x_ret

def output(x, nb_classes):
    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), padding='valid', name='conv10')(x)
    x = GlobalAveragePooling2D(name='avgpool10')(x)
    x = Activation("softmax", name='softmax')(x)
    return x