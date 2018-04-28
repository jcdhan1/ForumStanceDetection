# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 03:25:29 2018

@author: aca15jch
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
import reader, preprocess, random, collections
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


max_features = 20000

class LSTM_Network:
    def __init__(self):
        self.nn = None
    
    def fit(self, train_arrays, train_labels):
        del self.nn
        self.nn = Sequential()
        self.nn.add(Embedding(max_features, 128,input_length = train_arrays.shape[1]))
        self.set_layer()
        self.nn.add(Dropout(0.5))
        self.nn.add(Dense(1,activation='sigmoid'))
        self.nn.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.nn.fit(train_arrays, train_labels, epochs = 8, shuffle=True)

class Conditional_Encoding(LSTM_Network):
    def __init__(self):
        super(Conditional_Encoding, self).__init__()
    
    def set_layer(self):
        self.nn.add(LSTM(128))

class Bidirectional_Encoding(LSTM_Network):
    def __init__(self):
        super(Bidirectional_Encoding, self).__init__()
    
    def set_layer(self):
        self.nn.add(Bidirectional(LSTM(128)))

def equal_stance_proportions(debate):
    #Get most common stance and its frequency and least frequent stance and its frequency
    most_tup, least_tup = collections.Counter([p.label for p in debate.post_list]).most_common()[:2]
    #Select all of the least common
    least_lst = list(filter(lambda p: p.label==least_tup[0], debate.post_list))
    #Randomly select n of the formerly most common where n is the frequency of what was least common.
    most_lst = list(filter(lambda p: p.label==most_tup[0], debate.post_list))
    random.shuffle(most_lst)
    most_lst = most_lst[0:least_tup[1]]
    equal_stances =  most_lst + least_lst
    random.shuffle(equal_stances)
    return preprocess.Debate(debate.topic, equal_stances)

if __name__ == '__main__':
    STANCES = {'AGAINST': 0, 'FAVOR': 1}
    #Get raw data
    rdr=reader.Reader('../data/CreateDebate/')
    #Make sure the proportions of each stance in training and testing data are equal
    train_posts = equal_stance_proportions(rdr.load_cd('obama', 'ALL')).post_list
    
    #Tokenisation of training data
    the_tokenizer = Tokenizer(num_words=max_features, split=' ')
    the_tokenizer.fit_on_texts(([p.body for p in train_posts])) #Only do this for the training data
    train_array = the_tokenizer.texts_to_sequences([p.body for p in train_posts])
    train_array = pad_sequences(train_array)
    train_label = [STANCES[p.label] for p in train_posts]
    
    #Training
    the_classifier = Bidirectional_Encoding()
    the_classifier.fit(train_array, train_label)
    
    #Tokenisation of testing data
    test_posts =  equal_stance_proportions(rdr.load_cd('marijuana', 'ALL')).post_list
    test_array = the_tokenizer.texts_to_sequences([p.body for p in test_posts])
    test_array = pad_sequences(test_array, maxlen=train_array.shape[1])
    test_label = [STANCES[p.label] for p in test_posts]
    
    #Testing
    score, acc = the_classifier.nn.evaluate(test_array, test_label)
    predictions = list(the_classifier.nn.predict_classes(test_array).flatten())
