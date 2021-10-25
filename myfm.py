#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 21:51:05 2021

@author: Neil
"""
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Dense,Add,Activation,Layer
from tensorflow.keras.models import Model

class IntersectLayer(Layer):
    def __init__(self, input_dim, K=10, **kwargs):
        self.input_dim = input_dim
        self.K = K
        super(IntersectLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # 创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.input_dim, self.K),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(IntersectLayer, self).build(input_shape)
        
    def call(self, x):
        # 实现公式
        sum_then_square = K.pow(K.dot(x, self.kernel), 2)
        square_then_sum = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return 0.5 * K.sum(sum_then_square - square_then_sum, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

def build_fm(input_dim=None, K=10):
    input_layer = Input(shape=(input_dim,))
    lr = Dense(1, kernel_initializer='normal')(input_layer)
    intersect = IntersectLayer(input_dim, K)(input_layer)
    merge = Add()([lr, intersect])
    output = Activation('sigmoid')(merge)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()
    return model

if __name__ == '__main__':
    dataset = load_breast_cancer() #这里找一个数据集只是为了让例子跑起来，用连续特征直接喂给FM的效果是不好的，需要先做离散化
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.15,
                                                        random_state=30, stratify=dataset.target)
    fm = build_fm(30, 10)
    
    fm.fit(X_train, y_train, batch_size=1024, epochs=300)
    
    y_pred = fm.predict(X_test)
    print('y_pred',y_pred)
    print('test auc', roc_auc_score(y_test, y_pred))
