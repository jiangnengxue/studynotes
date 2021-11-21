#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:20:46 2021
@author: Neil
"""

import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Embedding, Dense, Flatten
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split

def data_process():
    # 采用CTR比赛的kaggle数据集
    data = pd.read_csv('./data/train_data')
    
    # 全部列名
    cols = ['C1',
            'banner_pos', 'site_domain', 'site_id', 'site_category',
            'app_id', 'app_category', 'device_type', 'device_conn_type',
            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
    # 连续特征
    continuous_col = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
    data.info()
    
    # 连续特征离散化处理
    dis = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data[continuous_col] = dis.fit_transform(data[continuous_col])
    
    # 数据标准化和归一化处理
    for f in cols:
        le = LabelEncoder()
        data[f] = le.fit_transform(data[f])
    
    def re_struct(feature_name, size, dim=8):
        return {'name': feature_name, 'size': size, 'dim': dim}
    
    feature_info = [re_struct(f, int(data[f].max()) + 1, 6) for f in cols]
    
    test_size = 0.2
    train_data, test_data = train_test_split(data, test_size=test_size)
    
    train_x = train_data[cols].values.astype(int)
    train_y = train_data['click'].values.astype(int)
    
    test_x = test_data[cols].values.astype(int)
    test_y = test_data['click'].values.astype(int)
    
    return train_x, train_y, test_x, test_y, feature_info

# 按域建Embedding表的方式
def build_embedding_layer(feature_info):
    emb_list = []
    for i, f in enumerate(feature_info):
        print(i, f, f['dim'])
        emb = Embedding(f['size'], f['dim'], input_length=1)
        emb_list.append(emb)
    return emb_list

def residual_block(hidden_dim, x_dim, inputs):
    layer1 = Dense(units=hidden_dim, activation='relu')
    layer2 = Dense(units=x_dim, activation=None)
    
    x = inputs
    x = layer1(x)
    x = layer2(x)
    outputs = tf.keras.activations.relu(inputs + x)
    return outputs

def build_residual_dnn():
    input_deep = tf.keras.layers.Input(shape=[input_length])
    emb_list = build_embedding_layer(feature_info)
    emb_input = tf.concat([emb_list[i](input_deep[:, i]) for i in range(input_deep.shape[1])], axis=-1)
    flatten_input = Flatten()(emb_input)
    
    block_layer = [128, 128, 128, 128, 128]
    x_dim = flatten_input.shape[1]
    h = flatten_input
    for dim in block_layer:
        h = residual_block(dim, x_dim, h)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    model = tf.keras.models.Model(inputs = [input_deep], outputs = [output])
    model.summary()
    return model

train_x, train_y, test_x, test_y, feature_info = data_process()
input_length = 17
model = build_residual_dnn()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC()])
history = model.fit(train_x, train_y, batch_size=1024, epochs=30,
                    use_multiprocessing=True, validation_split=0.1, verbose=1)
model.save('./save/dc.h5')

e = model.evaluate(test_x, test_y, batch_size=1024)
print(e)
