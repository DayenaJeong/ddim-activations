# Where do I change the activation function?
def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x) # try to change this layer
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply
#############################################################
# Sigmoid

#############################################################
# Softmax

#############################################################
# Tanh

#############################################################
# ReLU

#############################################################
# Leaky ReLU

x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
# Apply LeakyReLU activation
x = layers.LeakyReLU()(x)
#############################################################
# PReLU

from tensorflow.keras.layers import PReLU

x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
x = PReLU()(x)
#############################################################
# ELU

#############################################################
# Softplus

#############################################################
# Swish

x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x) # default
#############################################################
# Mish

import tensorflow as tf
from tensorflow.keras import layers

# Mish 활성화 함수 정의
def mish(x): 
    return x * tf.math.tanh(tf.math.softplus(x))

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        
        x = layers.BatchNormalization(center=False, scale=False)(x)
        # Conv2D의 activation을 mish로 변경
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=mish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply
