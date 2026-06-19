import tensorflow as tf
from tensorflow.keras import layers, Model

def stem_block(input_tensor):
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='valid', activation='relu')(input_tensor)
    x = layers.Conv2D(32, (3, 3), padding='valid', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.Conv2D(80, (1, 1), padding='valid', activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='valid', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x

def inception_A(input_tensor, scale=1.0):
    branch_0 = layers.Conv2D(96, (1, 1), padding='same', activation='relu')(input_tensor)

    branch_1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_1 = layers.Conv2D(96, (3, 3), padding='same', activation='relu')(branch_1)

    branch_2 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_2 = layers.Conv2D(96, (3, 3), padding='same', activation='relu')(branch_2)
    branch_2 = layers.Conv2D(96, (3, 3), padding='same', activation='relu')(branch_2)

    branch_3 = layers.Conv2D(96, (1, 1), padding='same', activation='relu')(input_tensor)

    output_tensor = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2, branch_3])
    output_tensor = layers.Lambda(lambda inputs, scale: inputs * scale, arguments={'scale': scale})(output_tensor)

    return output_tensor

def inception_B(input_tensor):
    branch_0 = layers.Conv2D(384, (1, 1), padding='same', activation='relu')(input_tensor)

    branch_1 = layers.Conv2D(192, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_1 = layers.Conv2D(224, (1, 7), padding='same', activation='relu')(branch_1)
    branch_1 = layers.Conv2D(256, (7, 1), padding='same', activation='relu')(branch_1)

    branch_2 = layers.Conv2D(192, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_2 = layers.Conv2D(192, (7, 1), padding='same', activation='relu')(branch_2)
    branch_2 = layers.Conv2D(224, (1, 7), padding='same', activation='relu')(branch_2)
    branch_2 = layers.Conv2D(224, (7, 1), padding='same', activation='relu')(branch_2)
    branch_2 = layers.Conv2D(256, (1, 7), padding='same', activation='relu')(branch_2)

    branch_3 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_3 = layers.Conv2D(192, (1, 7), padding='same', activation='relu')(branch_3)
    branch_3 = layers.Conv2D(192, (7, 1), padding='same', activation='relu')(branch_3)
    branch_3 = layers.Conv2D(224, (1, 7), padding='same', activation='relu')(branch_3)
    branch_3 = layers.Conv2D(224, (7, 1), padding='same', activation='relu')(branch_3)
    branch_3 = layers.Conv2D(256, (1, 7), padding='same', activation='relu')(branch_3)

    output_tensor = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2, branch_3])
    return output_tensor

def inception_C(input_tensor, scale=1.0):
    branch_0 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(input_tensor)

    branch_1 = layers.Conv2D(384, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_1a = layers.Conv2D(256, (1, 3), padding='same', activation='relu')(branch_1)
    branch_1b = layers.Conv2D(256, (3, 1), padding='same', activation='relu')(branch_1)

    branch_2 = layers.Conv2D(384, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_2 = layers.Conv2D(448, (3, 1), padding='same', activation='relu')(branch_2)
    branch_2 = layers.Conv2D(512, (1, 3), padding='same', activation='relu')(branch_2)
    branch_2a = layers.Conv2D(256, (1, 3), padding='same', activation='relu')(branch_2)
    branch_2b = layers.Conv2D(256, (3, 1), padding='same', activation='relu')(branch_2)

    branch_3 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(input_tensor)

    output_tensor = layers.Concatenate(axis=-1)([branch_0, branch_1a, branch_1b, branch_2a, branch_2b, branch_3])
    output_tensor = layers.Lambda(lambda inputs, scale: inputs * scale, arguments={'scale': scale})(output_tensor)

    return output_tensor

def reduction_A(input_tensor):
    branch_0 = layers.Conv2D(384, (3, 3), strides=(2, 2), padding='valid', activation='relu')(input_tensor)

    branch_1 = layers.Conv2D(192, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_1 = layers.Conv2D(224, (3, 3), padding='same', activation='relu')(branch_1)
    branch_1 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='valid', activation='relu')(branch_1)

    branch_2 = layers.MaxPooling2D((3, 3), strides=(2, 2))(input_tensor)

    output_tensor = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2])
    return output_tensor

def reduction_B(input_tensor):
    branch_0 = layers.Conv2D(192, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_0 = layers.Conv2D(192, (3, 3), strides=(2, 2), padding='valid', activation='relu')(branch_0)

    branch_1 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(input_tensor)
    branch_1 = layers.Conv2D(256, (1, 7), padding='same', activation='relu')(branch_1)
    branch_1 = layers.Conv2D(320, (7, 1), padding='same', activation='relu')(branch_1)
    branch_1 = layers.Conv2D(320, (3, 3), strides=(2, 2), padding='valid', activation='relu')(branch_1)

    branch_2 = layers.MaxPooling2D((3, 3), strides=(2, 2))(input_tensor)

    output_tensor = layers.Concatenate(axis=-1)([branch_0, branch_1, branch_2])
    return output_tensor

def InceptionV4(input_shape=(299, 299, 3), num_classes=1000):
    input_tensor = layers.Input(shape=input_shape)
    
    x = stem_block(input_tensor)
    
    # 4 x Inception-A
    for _ in range(4):
        x = inception_A(x)
    
    # Reduction-A
    x = reduction_A(x)
    
    # 7 x Inception-B
    for _ in range(7):
        x = inception_B(x)
    
    # Reduction-B
    x = reduction_B(x)
    
    # 3 x Inception-C
    for _ in range(3):
        x = inception_C(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.8)(x)
    output_tensor = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(input_tensor, output_tensor)
    return model
