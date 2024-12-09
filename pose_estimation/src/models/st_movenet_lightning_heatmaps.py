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
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, UpSampling2D, Activation, Add, BatchNormalization, ReLU
from tensorflow.keras.regularizers import L2


def mobileNetV2(shape=(192,192,3), alpha=1.0, pretrained_weights=None, trainable=True):

    backBone = tf.keras.applications.MobileNetV2(weights     = pretrained_weights,
                                                 alpha       = alpha, 
                                                 include_top = False, 
                                                 input_shape = shape)
    if not trainable:
        backBone.trainable = trainable
    return backBone

def st_movenet_lightning_heatmaps(input_shape, nb_keypoints, alpha, pretrained_weights, backbone_trainable=True):

    backbone = mobileNetV2(input_shape,alpha,pretrained_weights,backbone_trainable)

    conv_0 = Conv2D(24, kernel_size=1, padding='SAME', use_bias=False)(backbone.get_layer(name='block_2_add').output) #index = 19).output) # block_2_add
    conv_1 = Conv2D(32, kernel_size=1, padding='SAME', use_bias=False)(backbone.get_layer(name='block_5_add').output) #index = 37).output) # block_5_add
    conv_2 = Conv2D(64, kernel_size=1, padding='SAME', use_bias=False)(backbone.get_layer(name='block_9_add').output) #index = 61).output) # block_9_add

    conv_0 = BatchNormalization()(conv_0)
    conv_1 = BatchNormalization()(conv_1)
    conv_2 = BatchNormalization()(conv_2)

    x = Conv2D(64, kernel_size=1, padding='SAME', use_bias=False)(backbone.output)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2),interpolation='bilinear')(x)
    x = Add()([x,conv_2])

    x = DepthwiseConv2D(kernel_size=3, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=1, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size=(2, 2),interpolation='bilinear')(x)
    x = Add()([x,conv_1])

    x = DepthwiseConv2D(kernel_size=3, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(24, kernel_size=1, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(size=(2, 2),interpolation='bilinear')(x)
    x = Add()([x,conv_0])

    x = DepthwiseConv2D(kernel_size=3, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(24, kernel_size=1, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x_kptsHM = DepthwiseConv2D(kernel_size=3, padding='SAME', use_bias=False)(x)
    x_kptsHM = BatchNormalization()(x_kptsHM)
    x_kptsHM = Conv2D(96, kernel_size=1, padding='SAME', use_bias=False)(x_kptsHM)
    x_kptsHM = BatchNormalization()(x_kptsHM)
    x_kptsHM = ReLU()(x_kptsHM)

    x_kptsHM = Conv2D(nb_keypoints, kernel_size=1, padding='SAME', use_bias=False)(x_kptsHM)
    x_kptsHM = BatchNormalization()(x_kptsHM)
    outputs  = Activation('sigmoid')(x_kptsHM)

    model = Model(inputs=backbone.input, outputs=outputs, name= "st_movenet_lightning_heatmaps")

    return model