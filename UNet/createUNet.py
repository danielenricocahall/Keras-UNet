'''
Created on Feb 22, 2019

@author: daniel
'''
from keras.models import Model, Input
from keras.layers import Convolution2D, Activation, BatchNormalization,MaxPooling2D, concatenate
from keras.layers.convolutional import UpSampling2D

def createUNet(input_shape = (128,128,1), 
               n_labels = 1, 
               numFilters = 32, 
               output_mode="sodtmax"):
    
    inputs = Input(input_shape)
        
    conv1 = Convolution2D(numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Convolution2D(2*numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Convolution2D(4*numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Convolution2D(8*numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Convolution2D(16*numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    
    up6 = UpSampling2D(size=(2,2))(conv5)
    conv6 = Convolution2D(8*numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    merge6 = concatenate([conv4,conv6],axis=3)
    
    up7 = UpSampling2D(size=(2,2))(merge6)
    conv7 = Convolution2D(4*numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    merge7 = concatenate([conv3,conv7],axis=3)
    
    up8 = UpSampling2D(size=(2,2))(merge7)
    conv8 = Convolution2D(2*numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    merge8 = concatenate([conv2,conv8],axis=3)

    up9 = UpSampling2D(size=(2,2))(merge8)
    conv9 = Convolution2D(numFilters, (3,3), padding='same', kernel_initializer = 'he_normal')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    merge9 = concatenate([conv1,conv9],axis=3)
    
    conv10 = Convolution2D(n_labels, (1,1),  padding = 'same',  kernel_initializer = 'he_normal')(merge9)
    conv10 = BatchNormalization()(conv10)
    outputs = Activation(output_mode)(conv10)
    
    model = Model(input = inputs, output = outputs)
 
    return model