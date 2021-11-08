#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 19:37:33 2021

@author: Neil
"""
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

def pre_text_process(text):
    words = text.split(' ')

    id_from_word = {}
    word_from_id = {}
    
    for w in words:
        if w not in id_from_word:
            id = len(id_from_word)
            word_from_id[id] = w
            id_from_word[w] = id

    corpus = np.array([id_from_word[w] for w in words])

    return corpus, id_from_word, word_from_id

def create_samples(corpus, window_size=1):
    y = corpus[window_size : -window_size]
    samples = []

    for i in range(window_size, len(corpus) - window_size):
        temp_list = []
        for offset in range(-window_size, window_size + 1):
            if offset != 0:
                temp_list.append(corpus[i + offset])
        samples.append(temp_list)

    return np.array(samples), np.array(y)

#计算余弦相似度
def cosine_similarity(a, b):
    e = 1e-10 #分母加上一个微小的数，防止除零错误
    A = a / (np.sqrt(np.sum(a * a)) + e)
    B = b / (np.sqrt(np.sum(b * b)) + e)
    return np.dot(A, B)

text = '我 是 中国 人 ， 他 是 英国 人 。' #以空格分词
corpus, id_from_word, word_from_id = pre_text_process(text)

x, y = create_samples(corpus, window_size=1)

unique_corpus = list(set(corpus))
vocab_size = len(unique_corpus)

corpus_one_hot = keras.utils.to_categorical(unique_corpus, vocab_size)

x_train = []
y_train = []
for i in range(y.size):
    x_train.append(corpus_one_hot[x[i][0]])
    x_train.append(corpus_one_hot[x[i][1]])
    y_train.append(corpus_one_hot[y[i]])

x = np.array(x_train)
x = x.reshape(y.size, 2, vocab_size)

y = np.array(y_train)

input_x = tf.keras.layers.Input(shape=(2, vocab_size), name='layer0')
flatten = tf.keras.layers.Flatten()(input_x)
hidden1 = tf.keras.layers.Dense(3, activation='relu', use_bias=False, name='layer1')(flatten)
output = tf.keras.layers.Dense(vocab_size, activation='softmax', use_bias=False, name='layer2')(hidden1)

simple_model = tf.keras.models.Model(inputs = [input_x], outputs = [output])
simple_model.summary()

simple_model.compile(optimizer='adam', loss='categorical_crossentropy')
simple_model.fit(x, y, epochs=1000, batch_size=1, verbose=2)

w = simple_model.get_weights()
print(w)

print(cosine_similarity(w[0][0], w[0][1]))
print(cosine_similarity(w[0][0], w[0][5]))
