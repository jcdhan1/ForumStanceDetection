# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 19:38:20 2018

@author: aca15jch
"""
import numpy as np
import reader, writer, preprocess, os
from twokenize_wrapper.twokenize import tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import scale

class Model_Wrapper:
    def __init__(self, filepath, train_data, test_data):
        self.filepath = filepath
        self.model = Word2Vec.load(filepath)
        self.stance_dict = {'AGAINST': -1, 'NONE': 0, 'FAVOR': 1}
    
    def vectorise_body(self, tokenised_body):
        v_of_vectors = np.zeros(self.num_features).reshape((1, self.num_features))
        words = 0
        for token in tokenised_body:
            try:
                v_of_vectors += self.model[token].reshape((1, self.num_features))
                words += 1
            except KeyError:
                continue
        if words>0:
            v_of_vectors /= words
        return v_of_vectors
    
    def vectorise_debate(self, debate):
        return scale(np.concatenate(list(map(lambda p: self.vectorise(writer.filterStopwords(tokenize(p.body.lower()))),debate.post_list))))
    
    def stance_to_int(self, debate):
        return list(map(lambda p: self.stance_dict[p.label], debate.post_list))
    
class Experiment:
    def __init__(self, classifier, img_path, dir_cd):
        """
        Constructor for an Experiment.
        
        :param classifier: the classifier, can be an instance svm.LinearSVC, sklearn.svm.SVC or one of the LSTM subclasses.
        :param img_path  : where to export the evaluation graphs.
        :param dir_cd    : the directory of the CreateDebate files (training data).
        """
        self.classifier = classifier
        self.img_path = img_path
        self.dir_cd = dir_cd

class Experiment1(Experiment):
    def __init__(self, classifier, img_path, dir_cd, topic):
        """
        A subclass of Experiment specifically for experiment setup 1.
        
        :param topic: The topic of all posts in the data-set.
        """
        super(Experiment1, self).__init__(classifier,img_path,img_path, dir_cd)
        self.topic = topic

class Experiment2(Experiment):
    def __init__(self, classifier, img_path, dir_cd, dir_4f, seen_target, unseen_target):
        """
        A subclass of Experiment specifically for experiment setup 2.
        
        :param dir_4f       : the directory of the 4Forum.com files.
        :param seen_target  : the topic of the training data.
        :param unseen_target: the topic of the testing data.
        """
        super(Experiment2, self).__init__(classifier,img_path,img_path, dir_cd)
        self.dir_4f = dir_4f
        self.seen_target = seen_target
        self.unseen_target = unseen_target
    
if __name__ == '__main__':
    #Experiment 1:
    dir_cd = input("Where are the posts from CreateDebate stored?")
    topic = reader.select_topic(dir_cd)
    