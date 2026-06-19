import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, Concatenate, Add, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D, AveragePooling2D, Input, Flatten
from keras.models import Model

def inception_stem(input):
    # Initial stem layer
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(32, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = Conv2D(80, (1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(192, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    return x

def inception_A(input):
    # Inception-A block
    branch_1 = Conv2D(64, (1, 1), padding='same')(input)
    branch_1 = BatchNormalization()(branch_1)
    branch_1 = Activation('relu')(branch_1)
    
    branch_2 = Conv2D(48, (1, 1), padding='same')(input)
    branch_2 = BatchNormalization()(branch_2)
    branch_2 = Activation('relu')(branch_2)
    branch_2 = Conv2D(64, (5, 5), padding='same')(branch_2)
    branch_2 = BatchNormalization()(branch_2)
    branch_2 = Activation('relu')(branch_2)
    
    branch_3 = Conv2D(64, (1, 1), padding='same')(input)
    branch_3 = BatchNormalization()(branch_3)
    branch_3 = Activation('relu')(branch_3)
    branch_3 = Conv2D(96, (3, 3), padding='same')(branch_3)
    branch_3 = BatchNormalization()(branch_3)
    branch_3 = Activation('relu')(branch_3)
    branch_3 = Conv2D(96, (3, 3), padding='same')(branch_3)
    branch_3 = BatchNormalization()(branch_3)
    branch_3 = Activation('relu')(branch_3)
    
    branch_4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_4 = Conv2D(32, (1, 1), padding='same')(branch_4)
   
