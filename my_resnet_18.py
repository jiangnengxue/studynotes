#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 19:25:12 2021

@author: Neil
"""
from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import keras
import math

def conv_block(x, filter_num, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filter_num, kernel_size = kernel_size, strides = strides,
                          padding = padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def shortcut_connection(x, block):
    x_shape = K.int_shape(x)
    block_shape = K.int_shape(block)
    
    stride_height = int(round(x_shape[1] / block_shape[1]))
    stride_width = int(round(x_shape[2] / block_shape[2]))
    x_channel = x_shape[3]
    block_channel = block_shape[3] 
    
    if(stride_height <= 1 and stride_width <= 1 and x_channel == block_channel): #不需要调整
        return add([x, block])
    
    #使用1x1卷积调整，使其维度相等
    x = Conv2D(filters = block_channel,
                    kernel_size = (1, 1),
                    strides = (stride_width, stride_height),
                    padding = "valid")(x)
    
    return add([x, block])
 
def basic_block(x, filter_num, strides=(1, 1)):
    conv1 = conv_block(x, filter_num, kernel_size=(3, 3), strides = strides)
    residual = conv_block(conv1, filter_num, kernel_size=(3, 3), strides = strides)
    
    return shortcut_connection(x, residual)

def residual_block(x, filter_num, repetitions):
    for i in range(repetitions):
        strides = (1, 1)
        x = basic_block(x, filter_num, strides)
    return x
 
def build_resnet_18(input_shape=(32, 32, 3), classes = 10):
    x = Input(shape = input_shape)
    conv1 = conv_block(x, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
    conv2 = residual_block(pool1, 64, 2)
    conv3 = residual_block(conv2, 128, 2)
    conv4 = residual_block(conv3, 256, 2)
    conv5 = residual_block(conv4, 512, 2)
    pool2 = GlobalAvgPool2D()(conv5)
    y = Dense(classes, activation = 'softmax')(pool2)
    model = Model(inputs = x, outputs = y)
    model.summary()
 
    return model

def step_decay(epoch):
    init_learning_rate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    learning_rate = init_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return learning_rate
    
if __name__ == '__main__':
    num_classes = 10
    img_rows = 32
    img_cols = 32
    img_channels = 3
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32') /255
    x_test = x_test.astype('float32')/ 255

    # 创建模型
    model = build_resnet_18(input_shape=(img_rows, img_cols, img_channels), classes = num_classes)
    plot_model(model, 'resnet_18-18.png')
    
    model.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics=['accuracy'])

    # 设置图像读取和预处理
    img_gen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 fill_mode='constant',cval=0.)
    img_gen.fit(x_train)

    # 设置回调
    layers = 8 * 2 + 2
    cbks = [TensorBoard(log_dir='./resnet_{:d}/'.format(layers), histogram_freq=0), LearningRateScheduler(step_decay)]
    
    # 开始训练
    batch_size = 128
    epochs = 30
    iterations = 30000

    model.fit_generator(img_gen.flow(x_train, y_train, batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
    model.save('model_resnet_{:d}.h5'.format(layers))
