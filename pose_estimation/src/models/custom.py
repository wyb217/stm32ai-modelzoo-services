# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, UpSampling2D, Activation, Add, BatchNormalization, ReLU, MaxPooling2D
from tensorflow.keras.regularizers import L2

def custom(input_shape, nb_keypoints):

    inputs = Input(shape=input_shape)

    # Define the feature extraction layers
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(nb_keypoints, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    outputs  = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name= "custom")

    return model