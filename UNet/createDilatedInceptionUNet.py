'''
Created on Jan 19, 2019

@author: daniel
'''

from keras.models import Model, Input
from keras.layers import Convolution2D, Activation, BatchNormalization,MaxPooling2D, concatenate
from keras.layers.convolutional import UpSampling2D
from Inception.InceptionModule import DilatedInceptionModule


    
def createDilatedInceptionUNet(input_shape = (240,240,1), 
                        n_labels = 1, 
                        numFilters = 32, 
                        output_mode="softmax"):
    
    
    inputs = Input(input_shape)
        
    conv1 = DilatedInceptionModule(inputs, numFilters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = DilatedInceptionModule(pool1, 2*numFilters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = DilatedInceptionModule(pool2, 4*numFilters)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = DilatedInceptionModule(pool3, 8*numFilters)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = DilatedInceptionModule(pool4,16*numFilters)

    up6 = UpSampling2D(size=(2,2))(conv5)
    up6 = DilatedInceptionModule(up6, 8*numFilters)
    merge6 = concatenate([conv4,up6],axis=3)
    
    up7 = UpSampling2D(size=(2,2))(merge6)
    up7 = DilatedInceptionModule(up7, 4*numFilters)
    merge7 = concatenate([conv3,up7],axis=3)
    
    up8 = UpSampling2D(size=(2,2))(merge7)
    up8 = DilatedInceptionModule(up8, 2*numFilters)
    merge8 = concatenate([conv2,up8],axis=3)
    
    up9 = UpSampling2D(size=(2,2))(merge8)
    up9 = DilatedInceptionModule(up9, numFilters)
    merge9 = concatenate([conv1,up9],axis=3)
    
    conv10 = Convolution2D(n_labels, (1,1),  padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv10 = BatchNormalization()(conv10)
    outputs = Activation(output_mode)(conv10)
    
    model = Model(input = inputs, output = outputs)
 
    return model
