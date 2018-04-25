# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:28:40 2018

@author: aca15jch
"""
import preprocess, nltk, reader
from gensim.models import Word2Vec
from twokenize_wrapper.twokenize import tokenize

class Writer:
    def __init__(self, train_data):
        """
        A class for vectorising data from forum posts.
        """
        self.train_data = train_data
        self.model = None
    
    def skipgram(self, context, num_workers, num_features):
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
        return model_sg
        
class Writer_X1(Writer):
    def __init__(self, train_data, topic, prefix):
        """
        Subclass of Writer for experiment setup 1
        """
        super(Writer_X1, self).__init__(train_data)
        self.topic = topic
        self.prefix = prefix

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
    not_a_dbt = rdr.load_cd(seen_target, 'A', True)
    all_marijuana = rdr.load_cd(seen_target, "ALL")
    
    #Instantiate a Writer_X1
    ex1gen = Writer_X1(not_a_dbt, "marijuana", "A")
    ex1gen.skipgram(15,4,151)
    
    #Instantiate a Writer
    ex2gen = Writer(all_marijuana)
    ex2gen.skipgram(15,4,151)
