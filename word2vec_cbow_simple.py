#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 19:37:33 2021

@author: Neil
"""
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

#计算余弦相似度
def cosine_similarity(a, b):
    e = 1e-10 #分母加上一个微小的数，防止除零错误
    A = a / (np.sqrt(np.sum(a * a)) + e)
    B = b / (np.sqrt(np.sum(b * b)) + e)
    return np.dot(A, B)

text = '我是中国人，他是英国人。'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(corpus)

x, y = create_samples(corpus, window_size=1)
unique_corpus = list(set(corpus))
corpus_one_hot = keras.utils.to_categorical(unique_corpus, vocab_size)

x_train = []
y_train = []
for i in range(y.size):
    x_train.append(corpus_one_hot[x[i][0]])
    x_train.append(corpus_one_hot[x[i][1]])
    y_train.append(corpus_one_hot[y[i]])

x = np.array(x_train)
x = x.reshape(y.size, 2, 7)
y = np.array(y_train)

input_x = tf.keras.layers.Input(shape=(2, 7), name='layer0')
flatten = tf.keras.layers.Flatten()(input_x)
hidden1 = tf.keras.layers.Dense(3, activation='relu', use_bias=False, name='layer1')(flatten)
output = tf.keras.layers.Dense(7, activation='softmax', use_bias=False, name='layer2')(hidden1)

simple_model = tf.keras.models.Model(inputs = [input_x], outputs = [output])
simple_model.summary()
simple_model.compile(optimizer='adam', loss='categorical_crossentropy')
simple_model.fit(x2, y2, epochs=1000, batch_size=1, verbose=2)

w = simple_model.get_weights()
print(w)

print(cosine_similarity(w[0][0], w[0][1]))
print(cosine_similarity(w[0][0], w[0][4]))
