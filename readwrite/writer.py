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
        self.model = None
    
    def skipgram(self, context, num_workers, num_features, save=False):
        """
        Generate a skip-gram file.
        
        :param context     : context window size
        :param num_workers : the number of workers
        :param num_features: the number of features
        """
        #Tokenise the post bodies in the training data
        tokenized_posts = list(map(lambda p: filterStopwords(tokenize(p.body.lower())),self.train_data.post_list))
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
        self.model = model_sg
        if save:
            self.save_model(model_sg, '.sgv')
        else:
            return model_sg
        
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
        model.save(self.out_path + self.train_data.topic + '/' + self.prefix + extension)
        
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
        model.save(self.out_path + self.train_data.topic + extension)

def filterStopwords(tokenised_body):
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

if __name__ == '__main__':
    #Instantiate a Reader
    rdr = reader.Reader('../data/CreateDebate/', '../data/fourforums/')
    
    seen_target = "marijuana"
    unseen_target = "marijuana legalization"
    
    #Load posts from both data-sets
    a_dbt = rdr.load_cd(seen_target, 'A')
    not_a_dbt = rdr.load_cd(seen_target, 'A', True)
    all_marijuana = rdr.load_cd(seen_target, "ALL")
    rndm_dbt = rdr.load_4f(unseen_target)
    
    #Instantiate a Writer_X1 and generate an .sgv file
    ex1gen = Writer_X1(not_a_dbt, a_dbt, "../out/experiment_1/", "marijuana", "A")
    ex1gen.skipgram(15,4,151)
    
    #Instantiate a Writer_X2 and generate an .sgv file
    ex2gen = Writer_X2(all_marijuana, preprocess.Debate(unseen_target, [rndm_dbt]), "../out/experiment_2/")
    ex2gen.skipgram(15,4,151)
