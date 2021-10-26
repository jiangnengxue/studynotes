#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 20:20:46 2021

@author: jiangnengxue
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Embedding, Flatten
from tensorflow.keras.utils import plot_model
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.compat.v1.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def build_wide_and_deep(wide_dim, deep_dim):
    #多输入
    #wide 有wide_dim个特征
    input_wide = tf.keras.layers.Input(shape=[wide_dim])
    embedding_wide = Embedding(20000, 1, input_length=wide_dim)(input_wide)
    flatten_wide = tf.keras.layers.Flatten()(embedding_wide)
    
    #deep 有deep_dim个特征
    input_deep = tf.keras.layers.Input(shape=[deep_dim])
    embedding_deep = Embedding(20000, 12, input_length=deep_dim)(input_deep)
    flatten = tf.keras.layers.Flatten()(embedding_deep)
    hidden1 = tf.keras.layers.Dense(16, activation='relu')(flatten)
    hidden2 = tf.keras.layers.Dense(16, activation='relu')(hidden1)
    
    concat = tf.keras.layers.concatenate([flatten_wide, hidden2], axis=1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)
    model = tf.keras.models.Model(inputs = [input_wide, input_deep],
                               outputs = [output])
    model.summary()
    return model

data = pd.read_csv('train_data.csv')

cols = ['C1',
        'banner_pos',
        'site_domain',
        'site_id',
        'site_category',
        'app_id',
        'app_category',
        'device_type',
        'device_conn_type',
        'C14',
        'C15',
        'C16',
        'C17',
        'C18',
        'C19',
        'C20',
        'C21'
        ]
data.info()

y = data['click'].astype(int)
X = data[cols[0:]]

C1 = np.expand_dims(X['C1'], axis=1)
banner_pos = np.expand_dims(X['banner_pos'], axis=1)
site_domain = np.expand_dims(X['site_domain'], axis=1)
site_id = np.expand_dims(X['site_id'], axis=1)
site_category = np.expand_dims(X['site_category'], axis=1)
app_id = np.expand_dims(X['app_id'], axis=1)
app_category = np.expand_dims(X['app_category'], axis=1)
device_type = np.expand_dims(X['device_type'], axis=1)
device_conn_type = np.expand_dims(X['device_conn_type'], axis=1)
C14 = np.expand_dims(X['C14'], axis=1)
C15 = np.expand_dims(X['C15'], axis=1)
C16 = np.expand_dims(X['C16'], axis=1)
C17 = np.expand_dims(X['C17'], axis=1)
C18 = np.expand_dims(X['C18'], axis=1)
C19 = np.expand_dims(X['C19'], axis=1)
C20 = np.expand_dims(X['C20'], axis=1)
C21 = np.expand_dims(X['C21'], axis=1)

X_wide = np.hstack((C1, banner_pos, site_domain, site_id, site_category,
                    app_id, app_category, device_type, device_conn_type,
                    C14, C15, C16, C17, C18, C19, C20, C21))

model = build_wide_and_deep(17, 17)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', auroc])

n_epoch = 30
history = model.fit([X_wide, X], y, batch_size=1024, epochs=n_epoch, workers=2, use_multiprocessing=True)
model.save('model.h5')

plt.plot(range(n_epoch),history.history['loss'])
plt.show()
