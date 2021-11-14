#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:13:52 2021

本程序是一个学习用的Demo，展示一种Attention的实现方式

@author: Neil
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd

class AttentionLayer(Layer):
    def __init__(self, output_dim, query_size, key_size, **kwargs):
        self.output_dim = output_dim
        self.embed_size = 3
        self.query_size = query_size
        self.key_size = key_size
        self.vv = []
        self.qq = []
        self.kk = []
        self.aa = []
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_query = self.add_weight(name='w_query', 
                                      shape=(self.query_size, self.embed_size),
                                      initializer='uniform',
                                      trainable=True)
        
        self.w_key = self.add_weight(name='w_key', 
                                      shape=(self.key_size, self.embed_size),
                                      initializer='uniform',
                                      trainable=True)
        
        self.w_value = self.add_weight(name='w_value', 
                                      shape=(self.key_size, self.embed_size),
                                      initializer='uniform',
                                      trainable=True)
                
        super(AttentionLayer, self).build(input_shape)

    # 计算 item与p1, p2, p3之间的attention关系
    def call(self, item, p1, p2, p3):
        # 查找Embedding向量，即参数矩阵W
        q = tf.nn.embedding_lookup(self.w_query, item)
        
        k1 = tf.nn.embedding_lookup(self.w_key, p1)
        k2 = tf.nn.embedding_lookup(self.w_key, p2)
        k3 = tf.nn.embedding_lookup(self.w_key, p3)
        
        v1 = tf.nn.embedding_lookup(self.w_value, p1)
        v2 = tf.nn.embedding_lookup(self.w_value, p2)
        v3 = tf.nn.embedding_lookup(self.w_value, p3)
        
        # 降维后用于计算
        q = tf.squeeze(q)
        k1 = tf.squeeze(k1)
        k2 = tf.squeeze(k2)
        k3 = tf.squeeze(k3)
        v1 = tf.squeeze(v1)
        v2 = tf.squeeze(v2)
        v3 = tf.squeeze(v3)
        
        # 对应向量对应位置相乘
        a1 = q * k1
        a2 = q * k2
        a3 = q * k3
        
        # 累加向量各维度，与上面的乘法组合起来相当于点积Dot product
        a1 = tf.reduce_sum(a1, 1) 
        a1 = tf.expand_dims(a1, 1) # 扩维用于对其计算
        
        a2 = tf.reduce_sum(a2, 1)
        a2 = tf.expand_dims(a2, 1) 
        
        a3 = tf.reduce_sum(a3, 1)
        a3 = tf.expand_dims(a3, 1) 
        
        # 将上面的点积结果组合起来，形成新的向量，用于过softmax变换
        a = tf.concat([a1, a2, a3], axis=1)
        a = tf.nn.softmax(a)
        
        # 权重a1与向量v1相乘，输出c1
        a1 = a[:,0]
        a1 = tf.expand_dims(a1, 1)
        a1 = tf.concat([a1, a1, a1], axis=1)
        
        c1 = tf.multiply(a1, v1)
        c1 = tf.reduce_sum(c1, -1)
        
        # 权重a2与向量v2相乘，输出c2
        a2 = a[:,1]
        a2 = tf.expand_dims(a2, 1)
        a2 = tf.concat([a2, a2, a2], axis=1)
        
        c2 = tf.multiply(a2, v2)
        c2 = tf.reduce_sum(c2, -1)

        # 权重a3与向量v3相乘，输出c3
        a3 = a[:,2]
        a3 = tf.expand_dims(a3, 1)
        a3 = tf.concat([a3, a3, a3], axis=1)
        
        c3 = tf.multiply(a3, v3)
        c3 = tf.reduce_sum(c3, -1)
        
        # 整个attention的输出：c = c1 + c2 + c3
        c = tf.add(c1, c2)
        c = tf.add(c, c3)

        # 用于调试
        self.qq.append(c1)
        self.kk.append(c2)
        self.aa.append(a)  
        self.vv.append(c)
        
        return tf.reshape(c, [-1, self.output_dim])
    
    def get_vv(self):
        return self.vv
    def get_qq(self):
        return self.qq
    def get_kk(self):
        return self.kk
    def get_aa(self):
        return self.aa
    
    def compute_output_shape(self, input_shape):
        print('----', input_shape)
        return (input_shape[0], self.output_dim)
    
def build_model():
    input_x = tf.keras.layers.Input(shape=(2), dtype=tf.float32)
    input_item = tf.keras.layers.Input(shape=(1), dtype=tf.int32) #商品id
    prefer1_x = tf.keras.layers.Input(shape=(1), dtype=tf.int32) #用户偏好1
    prefer2_x= tf.keras.layers.Input(shape=(1), dtype=tf.int32) #用户偏好2
    prefer3_x = tf.keras.layers.Input(shape=(1), dtype=tf.int32) #用户偏好3
    
    attention = AttentionLayer(1, 8, 8, name="attention")(input_item, prefer1_x, prefer2_x, prefer3_x)
    merge = tf.keras.layers.concatenate([attention, input_x], axis=1)
    hidden2 = tf.keras.layers.Dense(16, activation='relu')(merge)
    hidden3 = tf.keras.layers.Dense(8, activation='relu')(hidden2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden3)
    
    model = tf.keras.models.Model(inputs = [input_x, input_item, prefer1_x, prefer2_x, prefer3_x], outputs = [output])
    model.summary()
    return model

# 读取数据和预处理
pd_data = pd.read_csv('./data/attention_train_data.csv')
pd_data.info()

data = pd_data[:-2]
test_data = pd_data[10:]

cols = ['user_id','age']
attention_item_cols = ['item_id']
prefer_cols = ['prefer1', 'prefer2', 'prefer3']

y = data['click'].astype(int)
x = data[cols].astype(float)
attention_item = data[attention_item_cols].astype(int)
prefer1 = data['prefer1'].astype(int)
prefer2 = data['prefer2'].astype(int)
prefer3 = data['prefer3'].astype(int)

# 定义模型
model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy')

# 开始训练
history = model.fit([x, attention_item, prefer1, prefer2, prefer3], y, epochs=500, batch_size=2)

# 调试信息
vv = model.get_layer("attention").get_vv()
qq = model.get_layer("attention").get_qq()
kk = model.get_layer("attention").get_kk()
aa = model.get_layer("attention").get_aa()

function = K.function(inputs=[model.get_input_at(0)], outputs=[qq, kk, aa, vv])
f1 = function([np.array(x), np.array(attention_item), np.array(prefer1), np.array(prefer2), np.array(prefer3)])
print(f1[0])
print('----')
print(f1[1])
print('----')
print(f1[2])
print('----')
print(f1[3])
print('----')

# 测试
y = test_data['click'].astype(int)
x = test_data[cols].astype(float)
attention_item = test_data[attention_item_cols].astype(int)
prefer1 = test_data['prefer1'].astype(int)
prefer2 = test_data['prefer2'].astype(int)
prefer3 = test_data['prefer3'].astype(int)
y_p = model.predict([np.array(x), np.array(attention_item), np.array(prefer1), np.array(prefer2), np.array(prefer3)])
print(np.round(y_p, 4))
