# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 19:38:20 2018
@author: aca15jch
"""

import numpy as np
import reader, writer, preprocess, copy, tabulate, argparse, conditional
from twokenize_wrapper.twokenize import tokenize
from sklearn import svm
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import random

import collections
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import re

STANCES = {'AGAINST': 0, 'FAVOR': 1}

class Model_Wrapper:
    def __init__(self, model, conditional=False):
        self.model = model                 
        self.conditional = conditional               
        if conditional:
            self.tokenizer =      Tokenizer(num_words=20000, split=' ')                                                                                                                                                                                                                                                  
    
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
        if self.conditional:
            
            sanitized_bodies = np.array([re.sub('[^a-zA-z0-9\s]','',p.body.lower()) for p in debate.post_list])
            self.tokenizer.fit_on_texts(sanitized_bodies) #only for training data
            vecs = self.tokenizer.texts_to_sequences(sanitized_bodies)
            vecs = pad_sequences(vecs)
            return vecs
        else:
            return scale(np.concatenate(list(map(lambda p: self.vectorise_body(writer.filterStopwords(tokenize(p.body.lower()))),debate.post_list))))

    
    def stance_to_n(self, debate):
        return [STANCES[p.label] for p in debate.post_list]
    
class Experiment:
    def __init__(self, classifier, img_path, dir_cd, train_data={}, test_data={}, models={}):
        """
        Constructor for an Experiment.
        
        :param classifier: the classifier, can be an instance svm.LinearSVC, sklearn.svm.SVC or one of the LSTM subclasses.
        :param img_path  : where to export the evaluation graphs.
        :param dir_cd    : the directory of the CreateDebate files (training data).
        """
        self.classifier = classifier
        self.img_path = img_path
        self.dir_cd = dir_cd
        self.test_data = test_data
        self.train_data = train_data
        self.models = models
        
    def img_directory(self):
        return self.topic
    
    def bar_plot(self, dict_1, dict_2, name_1, name_2, classifier, m_name=''):
        
        N = len(dict_1)
        val_1 = tuple(dict_1.values())
        
        ind = np.arange(N)
        width = 0.35
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, val_1, width, color='r')
        
        val_2 = tuple(dict_2.values())
        rects2 = ax.bar(ind + width, val_2, width, color='b')
        
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(tuple(dict_1.keys()))
        ax.set_ylim([0,1])
        ax.legend((rects1[0], rects2[0]), (name_1, name_2))
        
        plt.savefig(self.img_path + self.img_directory() + '/' + x1_filename(classifier,name_1,name_2, m_name), format='svg')
        plt.show()

class Experiment1(Experiment):
    def __init__(self, classifier, img_path, dir_cd, topic, train_data={}, test_data={}, models={}):
        """
        A subclass of Experiment specifically for experiment setup 1.
        
        :param topic: The topic of all posts in the data-set.
        """
        super(Experiment1, self).__init__(classifier, img_path, dir_cd, train_data, test_data, models)
        self.topic = topic
    
    def evaluate(self):
        mets = {}
        accs = {}
        max_props = {}
        for prefix in reader.subsetAZ(self.dir_cd + self.topic):
            wvmodel = self.models[prefix]
            
            
            if wvmodel.conditional:
                
            
                #Tokenisation of training data
                max_features = 20000
                the_tokenizer = Tokenizer(num_words=max_features, split=' ')
                the_tokenizer.fit_on_texts(([p.body for p in self.train_data[prefix].post_list])) #Only do this for the training data
                train_array = the_tokenizer.texts_to_sequences([p.body for p in self.train_data[prefix].post_list])
                train_array = pad_sequences(train_array)
                train_label = [STANCES[p.label] for p in self.train_data[prefix].post_list]
                
                #Training
                self.classifier.fit(train_array, train_label)
                
                #Tokenisation of testing data
                test_array = the_tokenizer.texts_to_sequences([p.body for p in self.test_data[prefix].post_list])
                test_array = pad_sequences(test_array, maxlen=train_array.shape[1])
                test_label = [STANCES[p.label] for p in self.test_data[prefix].post_list]
                
                
                score, accuracy = self.classifier.nn.evaluate(test_array, test_label)
                predicted = list(self.classifier.nn.predict_classes(test_array).flatten())
            else:
                train_arrays = wvmodel.vectorise_debate(self.train_data[prefix])
                test_arrays  = wvmodel.vectorise_debate(self.test_data[prefix])
                train_label = wvmodel.stance_to_n(self.train_data[prefix])
                test_label = wvmodel.stance_to_n(self.test_data[prefix])
                print("Training on", topic, "debates except for", prefix)
                self.classifier.fit(train_arrays,train_label)
                print("Testing on", topic, "debate", prefix)
                
                accuracy = self.classifier.score(test_arrays, test_label)
                predicted = list(self.classifier.predict(test_arrays))
            
            accs[prefix] = accuracy
            print("Accuracy:", accuracy)
           
            metrics={'Metric': ['Precision', 'Recall', 'F-measure', 'Proportion in Training Data']}
           
            
            print(test_label)
            print(predicted)
            print(set(test_label))
            print(set(predicted))
            print(len(test_label))
            print(len(predicted))
            
            proportions=[]
            for stance, n in STANCES.items():
                
                denominatorP=predicted.count(n)
                denominatorR=test_label.count(n)
                if denominatorP + predicted.count(abs(1-n)) !=len(test_label):
                    raise ValueError('Incorrect number of stances')
                
                #Avoid division by 0
                numerator = correctness(test_label,predicted, n)
                                    
                precision = 0
                if(denominatorP!=0):
                    precision = numerator/denominatorP
                    
                recall = 0
                if(denominatorR!=0):
                    recall = numerator/denominatorR
                
                f_measure = 0
                if(precision+recall!=0):
                    f_measure = 2*precision*recall/(precision+recall)
                
            
                proportion = train_label.count(n)/len(train_label)
                proportions+=[proportion]
                metrics[stance] = [precision, recall, f_measure, proportion]
            max_props[prefix] = max(proportions)
            mets[prefix] = metrics
        return mets, accs, max_props

def correctness(actual, predicted, class_n):
    return list(zip(actual, predicted)).count((class_n, class_n))

def extract_metrics(mets, stance, m_int):
    #m_int 0 for precision, 1 for recall, 2 for f-measure
    return dict(map(lambda m: (m[0], m[1][stance][m_int]), mets.items()))

def x1_filename(classifier, name_1, name_2, m_name=''):
    return classifier.replace(" ", "") + name_1 + name_2 + m_name + '.svg'

def equal_stance_proportions(debate):
    #Get most common stance and its frequency and least frequent stance and its frequency
    most_tup, least_tup = collections.Counter([p.label for p in debate.post_list]).most_common()
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
    plt.rcParams['svg.fonttype'] = 'none'
    parser = argparse.ArgumentParser(description='Run experiments setup 1 or 1.5.')
    parser.add_argument('--classifier', '-c', nargs='?', default ='', help='The classifier to use.')
    parser.add_argument('--image', '-i', nargs='?', default ='', help='Where to output graphs.')
    parser.add_argument('--setup', '-s', nargs='?', default = 0, type=int, help='Experiment setup to use.')
    parser.add_argument('--createdebate', '-d', nargs='?', default ='', help='The CreateDebate post directory.')
    parser.add_argument('--topic', '-t', nargs='?', default ='', help='The topic (setup 1) or seen target (setup 1.5)')
    #parser.add_argument('--4forums', '-4', nargs='?', default ='', help='The 4Forums.com post directory.')
    parser.add_argument('--unseen', '-u', nargs='?', default ='', help='The unseen target (setup 1.5).')
    args = vars(parser.parse_args())
    
    classifier_arr = [svm.LinearSVC()] + list(map(lambda k: svm.SVC(kernel=k), ['linear', 'poly', 'sigmoid', 'rbf'])) #Will implement LSTMs
    baselines = ['liblinear', 'linear', 'Polynomial','Sigmoid','Radial Basis Function']
    classifier_names = baselines + ['conditional', 'bidirectional']
    classifier_arr = classifier_arr + [conditional.Conditional_Encoding(), conditional.Bidirectional_Encoding()]
    classifier_dict = dict(zip(classifier_names, classifier_arr))
    valid_classifier = args['classifier'].lower() in classifier_names
    classifier_choice = args['classifier'].lower() if valid_classifier else reader.select_opt(classifier_names, "Select a classifier: ")

    dir_cd = input("Where are the posts from CreateDebate stored?") if not args['createdebate'] else args['createdebate'] #./data/CreateDebate/
    rdr=reader.Reader(dir_cd)
    
    valid_setup = 0 < args['setup'] < 3
    setup_choice = 'setup ' + str(args['setup']) if valid_setup else reader.select_opt(['setup 1','setup 1_5'],'Select an experiment setup:')
    img_path = input("Where should graphs be exported to?") if not args['image'] else args['image']
    topic = rdr.select_topic() if args['topic'] not in rdr.dir_lst else args['topic']
    
    
    
    if setup_choice == 'setup 1':
        #Experiment 1 args: -i ./img/experiment_1/ -s 1 -d ./data/CreateDebate/
        
        if(input('reload data for experiment 1?').lower() == 'y'):
            train_data1={}
            test_data1={}
            for prefix in reader.subsetAZ(dir_cd + topic):
                print('Loading', prefix)
                train_data1[prefix] = rdr.load_cd(topic, prefix, True)
                test_data1[prefix] = rdr.load_cd(topic, prefix)
        
        if classifier_choice in baselines:
            wvm_gen = writer.Writer_X1(train_data1[prefix], topic, prefix)
            if(input('regenerate models for experiment 1?') == 'y'):
                models1 = {}
                print('Generating Models')
                for prefix in reader.subsetAZ(dir_cd + topic):
                    print(prefix)
                    models1[prefix] =  Model_Wrapper(wvm_gen.skipgram(15,4,151))
                    models1[prefix].model.save('./out/experiment_1/' + topic + '/' + prefix + '.wv')
            else:
                print('Loading Models')
                for prefix in reader.subsetAZ(dir_cd + topic):
                    print(prefix)
                    models1[prefix] =  Model_Wrapper(Word2Vec.load('./out/experiment_1/' + topic + '/' + prefix + '.wv'))
        else:
            print('Generating Models')
            models1 = {}
            for prefix in reader.subsetAZ(dir_cd + topic):
                print(prefix)
                models1[prefix] = Model_Wrapper(None, True)
        
        
        classifier1 = copy.deepcopy(classifier_dict[classifier_choice]) 
        experiment1 = Experiment1(classifier1,img_path,dir_cd, topic, train_data1, test_data1, models1)
        mets, accs, max_props = experiment1.evaluate()
        print("Accuracy:", np.mean(list(accs.values())))
        for prefix, metrics in mets.items():
            print("\\newline")
            print(prefix)
            print('\\newline')
            print("Accuracy:", accs[prefix])
            print('\\newline')
            print(tabulate.tabulate(metrics, headers="keys", tablefmt="latex"))
        print('\\newline')
        print("Number of times accuracy was greater than proportion of most frequent stance in training data:", sum([accs[prefix] > max_props[prefix] for prefix in reader.subsetAZ(dir_cd + topic)]))
        print("\\newline")
        print("\\begin{minipage}{\linewidth}")
        print('    Accuracy')
        print('    \\newline')
        print("    \includesvg{img/experiment_1/" + topic + '/' + x1_filename(classifier_choice, 'accuracy','most-frequent-class') + "}")
        print("\end{minipage}")
        print("\\begin{minipage}{\linewidth}")
        print('    Precision')
        print('    \\newline')
        print("    \includesvg{img/experiment_1/" + topic + '/' + x1_filename(classifier_choice, 'AGAINST','FAVOR', 'precision') + "}")
        against_p = extract_metrics(mets, 'AGAINST', 0)
        favor_p = extract_metrics(mets, 'FAVOR', 0)
        print("\end{minipage}")
        print("\\begin{minipage}{\linewidth}")
        print('    Recall')
        print('    \\newline')
        print("    \includesvg{img/experiment_1/" + topic + '/' + x1_filename(classifier_choice, 'AGAINST','FAVOR', 'recall') + "}")
        against_r = extract_metrics(mets, 'AGAINST', 1)
        favor_r = extract_metrics(mets, 'FAVOR', 1)
        print("\end{minipage}")
        print("\\begin{minipage}{\linewidth}")
        print('    F-Measure')
        print('    \\newline')
        print("    \includesvg{img/experiment_1/" + topic + '/' + x1_filename(classifier_choice, 'AGAINST','FAVOR', 'f-measure') + "}")
        against_f = extract_metrics(mets, 'AGAINST', 2)
        favor_f = extract_metrics(mets, 'FAVOR', 2)
        print("\end{minipage}")
        experiment1.bar_plot(accs, max_props, 'accuracy','most-frequent-class', classifier_choice)
        experiment1.bar_plot(against_p, favor_p, 'AGAINST','FAVOR', classifier_choice, 'precision')
        experiment1.bar_plot(against_r, favor_r, 'AGAINST','FAVOR', classifier_choice, 'recall')
        experiment1.bar_plot(against_f, favor_f, 'AGAINST','FAVOR', classifier_choice, 'f-measure')
        
    else:
        #Experiment 1.5 args: -i ./img/experiment_1_5/ -s 2 -d ./data/CreateDebate/
        classifier1_5 = copy.deepcopy(classifier_dict[classifier_choice])
        unseen_target = rdr.select_topic() if args['unseen'] not in rdr.dir_lst else args['unseen']
        print("foo bar")
        if(input('Reload data?') == 'y'):
            train_data = rdr.load_cd(topic, 'ALL')
            test_data = rdr.load_cd(unseen_target, 'ALL')
            
            train_data = equal_stance_proportions(train_data)
            test_data = equal_stance_proportions(test_data)
        if classifier_choice in baselines:
            wvm_gen = writer.Writer(train_data)
            wvmodel = Model_Wrapper(wvm_gen.skipgram(15,4,151))
            test_arrays = wvmodel.vectorise_debate(test_data)
        else:
            wvmodel = Model_Wrapper(None,True)
            
        train_arrays = wvmodel.vectorise_debate(train_data)
        train_label = wvmodel.stance_to_n(train_data)
        test_label = wvmodel.stance_to_n(test_data)
        classifier1_5.fit(train_arrays, train_label)
        if classifier_choice in baselines:
            accuracy = classifier1_5.score(test_arrays, test_label)
            predicted = list(classifier1_5.predict(test_arrays))
        else:
            sanitized_bodies = np.array([re.sub('[^a-zA-z0-9\s]','',p.body.lower()) for p in test_data.post_list])
            
            #tokenizer.fit_on_texts(sanitized_bodies)
            vecs = wvmodel.tokenizer.texts_to_sequences(sanitized_bodies)
            test_arrays = pad_sequences(vecs, maxlen=train_arrays.shape[1], dtype='int32', padding='post', truncating='post', value=0)
            score,accuracy = classifier1_5.nn.evaluate(test_arrays, test_label, verbose = 2, batch_size = 64)
            print("score: %.2f" % (score))
            print("acc: %.2f" % (accuracy))
            predicted = list(classifier1_5.nn.predict_classes(test_arrays).flatten())
        
        
        print(accuracy)
        print(predicted)
        
        dummy = Experiment(None, img_path,dir_cd)
        dummy.topic = None
        metrics = {}
        for stance, n in STANCES.items():
            
            denominatorP=predicted.count(n)
            denominatorR=test_label.count(n)
            if denominatorP + predicted.count(abs(1-n)) !=len(test_label):
                raise ValueError('Incorrect number of stances')
            
            #Avoid division by 0
            numerator = correctness(test_label,predicted, n)
                                
            precision = 0
            if(denominatorP!=0):
                precision = numerator/denominatorP
                
            recall = 0
            if(denominatorR!=0):
                recall = numerator/denominatorR
            
            f_measure = 0
            if(precision+recall!=0):
                f_measure = 2*precision*recall/(precision+recall)
            
        
    
            metrics[stance] = [precision, recall, f_measure]
