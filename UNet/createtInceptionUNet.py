'''
Created on Oct 30, 2018

@author: daniel
'''

from keras.models import Model, Input
from keras.layers import Convolution2D, Activation, BatchNormalization,MaxPooling2D, concatenate
from keras.layers.convolutional import UpSampling2D
from Inception.InceptionModule import InceptionModule
    
def createInceptionUNet(input_shape = (128,128,1), 
                        n_labels = 1, 
                        numFilters = 32, 
                        output_mode="softmax"):
    
    inputs = Input(input_shape)
    
    conv1 = InceptionModule(inputs, numFilters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = InceptionModule(pool1, 2*numFilters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = InceptionModule(pool2, 4*numFilters)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = InceptionModule(pool3, 8*numFilters)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = InceptionModule(pool4,16*numFilters)

    up6 = UpSampling2D(size=(2,2))(conv5)
    up6 = InceptionModule(up6, 8*numFilters)
    merge6 = concatenate([conv4,up6],axis=3)
    
    up7 = UpSampling2D(size=(2,2))(merge6)
    up7 = InceptionModule(up7, 4*numFilters)
    merge7 = concatenate([conv3,up7],axis=3)
    
    up8 = UpSampling2D(size=(2,2))(merge7)
    up8 = InceptionModule(up8, 2*numFilters)
    merge8 = concatenate([conv2,up8],axis=3)
    
    up9 = UpSampling2D(size=(2,2))(merge8)
    up9 = InceptionModule(up9, numFilters)
    merge9 = concatenate([conv1,up9],axis=3)
    
    conv10 = Convolution2D(n_labels, (1,1),  padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv10 = BatchNormalization()(conv10)
    outputs = Activation(output_mode)(conv10)
    
    model = Model(input = inputs, output = outputs)
 
    return model
