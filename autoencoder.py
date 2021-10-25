#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 14:27:00 2021

@author: Neil
"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K

# 随机得到的输入数据
data = np.array(
        [[0.21, 0.41, 0.51],
        [0.92, 0.72, 0.90],
        [0.63, 0.13, 0.73],
        [0.78, 0.94, 0.34]])

# 构建一个AutoEncoder模型
def build_autoencoder(input_dim=None, bottleneck_dim=None):
    output_dim = input_dim
    input_layer = Input(shape=(input_dim,))
    # 编码层
    encoded = Dense(bottleneck_dim, activation='tanh', name='hidden')(input_layer)
    # 解码层
    decoded = Dense(output_dim, activation='tanh', name='output')(encoded)
    model = Model(inputs=input_layer, outputs=decoded)
    # 采用均方误差损失函数
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    return model
  
input_dim = 3
bottleneck_dim = 2
autoencoder = build_autoencoder(input_dim, bottleneck_dim)

# 训练模型
history = autoencoder.fit(data, data, epochs=10000)

# 预测结果并打印
y = autoencoder.predict(data)
print(data)
print(np.around(y, 2))

# 查看一下bottleneck层的数据长什么样
layer_n = K.function(inputs=[autoencoder.get_input_at(0)], 
                     outputs=[autoencoder.get_layer("hidden").output])
compressed_data = layer_n([data])
print(np.around(compressed_data, 2))
