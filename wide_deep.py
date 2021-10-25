#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 20:20:46 2021

@author: jiangnengxue
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
#import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Embedding, Flatten

from sklearn import preprocessing
from tensorflow.keras.utils import plot_model
