# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 03:25:29 2018

@author: aca15jch
"""
import keras.backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from functools import partial


def precision(class_n,y_true, y_pred):
    precision = 0
    print(np.array(y_true))
    print(np.array(y_pred))
    numerator = correctness(np.array(y_true).flatten(), np.array(y_pred).flatten(), class_n)
    denominator = list(np.array(y_pred).flatten()).count(class_n)
    if denominator> 0:
        precision = numerator/denominator
    return precision

def precision0(y_true, y_pred):
    return precision(0, y_true, y_pred)

def precision1(y_true, y_pred):
    return precision(1, y_true, y_pred)

def correctness(y_true, y_pred, class_n):
    return list(zip(y_true, y_pred)).count((class_n, class_n))

class LSTM_Network:
    def __init__(self):
        self.nn = None
    
    def predict_helper(self, vectorized_body):
        output_vec = self.nn.predict(np.array([vectorized_body]),batch_size=1,verbose = 2)[0]
        rounded = np.round(output_vec[0])
        return rounded
    
    def predict(self, test_arrays):
        lowest = 1
        predictions = []
        for vectorized_body in test_arrays:
            pred = self.nn.predict(np.array([vectorized_body]),batch_size=1,verbose = 2)[0][0]
            if pred < lowest:
                lowest = pred
            predictions += [pred]
        print(predictions)
        return [(0 if pred < (lowest+1)/2 else 1) for pred in predictions]
    
    def score(self, test_arrays, test_labels):
        numerator = [ls[0]==ls[1] for ls in (zip(self.predict(test_arrays),test_labels))].count(True)
        return numerator/test_arrays.shape[0]
    
    def fit(self, train_arrays, train_labels):
        del self.nn
        self.nn = Sequential()
        self.nn.add(Embedding(20000, 128,input_length = train_arrays.shape[1]))
        self.set_layer()
        self.nn.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.nn.fit(train_arrays, train_labels, epochs = 8, shuffle=True)
        
class Conditional_Encoding(LSTM_Network):
    def __init__(self):
        super(Conditional_Encoding, self).__init__()
    
    def set_layer(self):
        self.nn.add(LSTM(128))
        self.nn.add(Dropout(0.5))
        self.nn.add(Dense(1,activation='sigmoid'))
    
class Bidirectional_Encoding(LSTM_Network):
    def __init__(self):
        super(Bidirectional_Encoding, self).__init__()
    
    def set_layer(self):
        self.nn.add(Bidirectional(LSTM(64)))
        self.nn.add(Dense(256,activation='relu', kernel_initializer="random_normal"))
        self.nn.add(Dropout(0.5))
        self.nn.add(Dense(1,activation='sigmoid'))
