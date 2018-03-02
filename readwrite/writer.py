# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:28:40 2018

@author: aca15jch
"""

import preprocess, nltk, reader
from gensim import utils
from gensim.models import Word2Vec
from twokenize_wrapper.twokenize import tokenize

class Writer:
    def __init__(self, train_data, test_data, out_path):
        """
        A class for vectorising data from forum posts.
        
        """
        self.train_data = train_data
        self.test_data = test_data
        self.out_path = out_path
    
    def filterStopwords(self,tokenised_body):
        """
        Remove stopwords from a tokenised body
        :param tokenised_body: tokenised body
        :return: tokens without stopwords
        """
        stops = nltk.corpus.stopwords.words("english")
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!",  "?"])
        stops.extend(["rt", "#semst", "...", "thats", "im", "'s", "via"])
        stops = set(stops)
        return [w for w in tokenised_body if (not w in stops and not w.startswith("http"))]

    def tokenise_body(self, body):
        return self.filterStopwords(tokenize(body.lower()))
    
    def skipgram(self, context, num_workers, num_features):
        #Tokenise
        tokenized_posts = list(map(lambda p: self.tokenise_body(p.body),self.train_data.post_list))
        #Vectorise
        model_sg = Word2Vec(sentences=tokenized_posts,
                        size=num_features, 
                        window=context, 
                        negative=20,
                        iter=50,
                        seed=1000,
                        workers=num_workers,
                        sg=1)
        #Where the model is saved to depends on the experiment setup; which subclass the object is an instance of.
        self.save_model(model_sg, '.sgv')
        
class Writer_X1(Writer):
    def __init__(self, train_data, test_data, out_path, topic, prefix):
        """
        Subclass of Writer for experiment setup 1 where both training and testing data are
        sourced from the CreateDebate data-set.
        """
        if not all(map(lambda p: type(p).__name__=='Post_CD',train_data.post_list + test_data.post_list)):
            raise ValueError('All items in train_data.postlist and test_data.postlist but be an instance of preprocess.Post_CD.')
        super(Writer_X1, self).__init__(train_data, test_data, out_path)
        self.topic = topic
        self.prefix = prefix
    
    def save_model(self, model, extension):
        """
        Models for experiment setup 1 are saved in the directory experiment_1 and a subdirectory named after the topic. The prefix of the testing data is the filename.
        
        """
        model.save(self.out_path + 'experiment_1/' + self.train_data.topic + '/' + self.prefix + extension)
        
class Writer_X2(Writer):
    def __init__(self, train_data, test_data, out_path):
        """
        Subclass of Writer for experiment setup 2 where training data is from the
        CreateDebate data-set but testing data is from the 4Forums.com data-set.
        """
        if not (all(map(lambda p: type(p).__name__=='Post_CD',train_data.post_list)) and all(map(lambda p: type(p).__name__=='Post',test_data.post_list))):
            raise ValueError('All items in train_data.postlist must be an instance of preprocess.Post_CD. All items in test_data.postlist must be an instance of preprocess.Post.')
        super(Writer_X2, self).__init__(train_data, test_data, out_path)
    
    def save_model(self, model, extension):
        """
        Models for experiment setup 2 are saved in directory experiment_2 as a file named after the seen target.
        """
        model.save(self.out_path + 'experiment_2/' + self.train_data.topic + extension)

if __name__ == '__main__':
    #Instantiate a Reader
    reader1 = reader.Reader('../data/CreateDebate/', '../data/fourforums/')
    a_dbt = reader.load_cd(topic_dir="marijuana", prefix='A', exclude=False)
    not_a_dbt = reader.load_cd(topic_dir="marijuana", prefix='A', exclude=True)
    all_marijuana = reader.load_cd(topic_dir="marijuana", prefix="ALL")
    
    ex1gen = Writer_X1(not_a_dbt, a_dbt, "../out", "marijuana", "A")
    ex1gen.skipgram(15,4,151)
    ex2gen = Writer_X2(all_marijuana, preprocess.Debate("marijuana legalization", [reader.load_4f()]), "../out/")
    ex2gen.skipgram(15,4,151)
    