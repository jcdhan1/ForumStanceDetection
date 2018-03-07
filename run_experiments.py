# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 19:38:20 2018

@author: aca15jch
"""
import numpy as np
import reader, writer, preprocess, os, copy
from twokenize_wrapper.twokenize import tokenize
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.preprocessing import scale

class Model_Wrapper:
    def __init__(self, model):
        #self.filepath = filepath
        self.model = model #Word2Vec.load(filepath)
        self.stance_dict = {'AGAINST': -1, 'NONE': 0, 'FAVOR': 1}
    
    def vectorise_body(self, tokenised_body):
        v_of_vectors = np.zeros(self.model.layer1_size).reshape((1, self.model.layer1_size))
        words = 0
        for token in tokenised_body:
            try:
                v_of_vectors += self.model[token].reshape((1, self.model.layer1_size))
                words += 1
            except KeyError:
                continue
        if words>0:
            v_of_vectors /= words
        return v_of_vectors
    
    def vectorise_debate(self, debate):
        return scale(np.concatenate(list(map(lambda p: self.vectorise_body(writer.filterStopwords(tokenize(p.body.lower()))),debate.post_list))))
    
    def stance_to_int(self, debate):
        return list(map(lambda p: self.stance_dict[p.label], debate.post_list))
    
class Experiment:
    def __init__(self, classifier, out_path, img_path, dir_cd):
        """
        Constructor for an Experiment.
        
        :param classifier: the classifier, can be an instance svm.LinearSVC, sklearn.svm.SVC or one of the LSTM subclasses.
        :param out_path  : where to export files for word vector models.
        :param img_path  : where to export the evaluation graphs.
        :param dir_cd    : the directory of the CreateDebate files (training data).
        """
        self.classifier = classifier
        self.out_path = out_path
        self.img_path = img_path
        self.dir_cd = dir_cd

class Experiment1(Experiment):
    def __init__(self, classifier, out_path, img_path, dir_cd, topic):
        """
        A subclass of Experiment specifically for experiment setup 1.
        
        :param topic: The topic of all posts in the data-set.
        """
        super(Experiment1, self).__init__(classifier, out_path, img_path, dir_cd)
        self.topic = topic
    
    def run(self, rdr):
        #Test on one debate, train on every other debate. Loop for all debates for the given topic.
        for prefix in reader.subsetAZ(self.dir_cd + self.topic):
            print("Generating model")
            train_data = rdr.load_cd(topic, prefix, True)
            test_data = rdr.load_cd(topic, prefix)
            wvm_gen = writer.Writer_X1(train_data, test_data, self.out_path, self.topic, prefix)
            wvmodel = Model_Wrapper(wvm_gen.skipgram(15,4,151))
            train_arrays = wvmodel.vectorise_debate(train_data)
            test_arrays  = wvmodel.vectorise_debate(test_data)
            train_labels = wvmodel.stance_to_int(train_data)
            test_labels = wvmodel.stance_to_int(test_data)
            print("Training on", topic, "debates except for", prefix)
            self.classifier.fit(train_arrays,train_labels)
            print("Testing on", topic, "debate", prefix)
            print("Accuracy:", self.classifier.score(test_arrays, test_labels))

class Experiment2(Experiment):
    def __init__(self, classifier, out_path, img_path, dir_cd, dir_4f, seen_target, unseen_target):
        """
        A subclass of Experiment specifically for experiment setup 2.
        
        :param dir_4f       : the directory of the 4Forum.com files.
        :param seen_target  : the topic of the training data.
        :param unseen_target: the topic of the testing data.
        """
        super(Experiment2, self).__init__(classifier, out_path, img_path, dir_cd)
        self.dir_4f = dir_4f
        self.seen_target = seen_target
        self.unseen_target = unseen_target
    
if __name__ == '__main__':
    classifiers = [svm.LinearSVC()] + list(map(lambda k: svm.SVC(kernel=k), ['linear', 'poly', 'sigmoid', 'rbf'])) #Will implement LSTMs
    classifier_names = ['liblinear', 'linear', 'Polynomial','Sigmoid','Radial Basis Function']
    classifier_dict = dict(zip(classifier_names, classifiers))
    

    dir_cd = './data/CreateDebate/' #input("Where are the posts from CreateDebate stored?") #./data/CreateDebate/
    dir_4f = './data/fourforums/' #input("Where are the posts from 4Forums.com stored?") #./data/fourforums/
    rdr = reader.Reader(dir_cd, dir_4f)
    
    if reader.select_opt(['setup 1','setup 2'],'Select an experiment setup:') == 'setup 1':
        #Experiment 1
        out_path = './out/experiment_1/' #input("Where should the files for word vector models be exported to?") #./out/experiment_1/
        img_path = './img/experiment_1/' #input("Where should graphs be exported to?") #./img/experiment_1/
        topic = reader.select_topic(dir_cd)
        #New instance of one of the classifier classes in the dictionary
        classifier1 = copy.deepcopy(classifier_dict[reader.select_opt(classifier_names, "Select a classifier:")]) 
        experiment1 = Experiment1(classifier1,out_path,img_path,dir_cd,topic)
        experiment1.run(rdr)
    else:
        #Experiment 2
        out_path = './out/experiment_2/' #input("Where should the files for word vector models be exported to?") #./out/experiment_2/
        img_path = './img/experiment_2/' #input("Where should graphs be exported to?") #./img/experiment_2/
        seen_target = reader.select_topic(dir_cd)
        unseen_target = rdr.select_target()
        classifier2 = copy.deepcopy(classifier_dict[reader.select_opt(classifier_names, "Select a classifier:")]) 
        experiment2 = Experiment2(classifier2,out_path,img_path,dir_cd,dir_4f,seen_target,unseen_target)
