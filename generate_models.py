"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) (directory, debate identifier or topic)
OPTIONS:
    --help   , -h       : print this help message
    --targets, -t       : If absent, the targets are unseen
    --mode=topic/post   : In topic mode (the default), targets are always the topic. In post mode, targets can be posts ids.
    --path=, -p path    : If present, this is the directory and the first argument is the topic of the debates.
    --topic=, -T Topic  : This is the target of the opening posts.
------------------------------------------------------------
"""
import os, sys, getopt, re, random, nltk
from readwrite import reader
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from twokenize_wrapper.twokenize import tokenize

def filterStopwords(tokenised_tweet, filter="all"):
    """
    Remove stopwords from tokenised tweet
    :param tokenised_tweet: tokenised tweet
    :return: tweet tokens without stopwords
    """
    if filter == "all":
        stops = nltk.corpus.stopwords.words("english")
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!",  "?"])
        stops.extend(["rt", "#semst", "...", "thats", "im", "'s", "via"])
    elif filter == "most":
        stops = []
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?"])
        stops.extend(["rt", "#semst", "...", "thats", "im", "'s", "via"])
    elif filter == "punctonly":
        stops = []
        # extended with string.punctuation and rt and #semst, removing links further down
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                  ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~"])  #"=", "+", "!",  "?"
        stops.extend(["rt", "#semst", "..."]) #"thats", "im", "'s", "via"])
    else:
        stops = ["rt", "#semst", "..."]

    stops = set(stops)
    return [w for w in tokenised_tweet if (not w in stops and not w.startswith("http"))]

def tokenise_body(body, stopwords="all"):
    return filterStopwords(tokenize(body.lower()), stopwords)

class Models_Generator:
    def __init__(self, config, num_features=175):
        self.config = config
        self.num_features = num_features  # Word vector dimensionality
        
        """Although printing the set will show the letters sorted, the letters are not accessed in order when
        #the set's converted to a list or used in a for-loop hence it must be converted to a list and then sorted."""
        subsetAZ = list(set(map(lambda f: f[0], os.listdir(config.filepath))))
        subsetAZ.sort()
        self.subsetAZ = subsetAZ
        self.skipgram_labels = dict.fromkeys(subsetAZ)
    
    def generate(self):
        print('Generating Distributed Memory and Skip-Gram model files for', len(self.subsetAZ), 'debates.')
        for prefix in self.subsetAZ:
            #Source, debate to use as testing-data, word vector dimensionality, window size, Number of threads to run in parallel
            vec_args = (prefix, 15, 4)  
            self.distributed_memory(*vec_args)
            self.skipgram(*vec_args)
    

    def build_tokenized_posts_list(self, debate_list, is_testing=False, is_doc2vec=True):
        tagged_posts_list=[]
        for p in debate_list:
            tokenised_body = tokenise_body(p.body)
            if is_doc2vec:
                tagged_posts_list.append(TaggedDocument(tokenised_body, 
                                                        [('TEST' if is_testing else 'TRAIN') + '_' + p.label + '_' + p.post_id]))
            else:
                tagged_posts_list.append(tokenised_body)
                
        return tagged_posts_list

    def distributed_memory(self, prefix, context, num_workers):
        train_debates =  reader.Debate(self.config.filepath, prefix, self.config.topic, True)
        test_debate = reader.Debate(self.config.filepath, prefix, self.config.topic)
        tagged_posts_list=self.build_tokenized_posts_list(test_debate.post_list, True) + self.build_tokenized_posts_list(train_debates.post_list)
    
        #Vectorization of Data 
        min_word_count = 5  # Minimum word count
        downsampling = 1e-3 # Downsample setting for frequent words
        #Doc2Vec by default uses "Distributed Memory".
        model_dm = Doc2Vec(min_count=min_word_count, window=context, size=self.num_features, sample=downsampling, workers=num_workers)
        model_dm.build_vocab(tagged_posts_list)
        for epoch in range(len(self.subsetAZ)*2): 
            shuffled=tagged_posts_list
            random.shuffle(shuffled)
            model_dm.train(shuffled, total_examples=model_dm.corpus_count, epochs=model_dm.iter)
        print('Saving DM model for when', prefix, 'debates are the testing data.')
        model_dm.save(out_path + '/' + prefix + '.dmv') #distributed memory vector file

    def skipgram(self, prefix, context, num_workers):
        train_debates =  reader.Debate(self.config.filepath, prefix, self.config.topic, True)
        #When the debate identified by the prefix is the testing data, generate a model based on debates identified by every other prefix.
        tokenized_debates= self.build_tokenized_posts_list(train_debates.post_list, False, False)
        model_sg = Word2Vec(sentences=tokenized_debates,
                        size=self.num_features, 
                        window=context, 
                        negative=20,
                        iter=50,
                        seed=1000,
                        workers=num_workers,
                        sg=1)
        print('Saving Skip-Gram model for when', prefix, 'debates are the testing data.')    
        model_sg.save(out_path + '/' + prefix + '.sgv') #skip-gram vector file

if __name__ == '__main__':
    #Example: ./data/CreateDebate/obama Obama
    
    #Reading of Data
    opts, args = getopt.getopt(sys.argv[1:],'htp:T:',["help", "targets", "mode=", "path=", "topic="])
    opts = dict(opts)
    # HELP option
    if '-h' in opts or '--help' in opts:
        help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
        print(help,file=sys.stderr)
        sys.exit()
    config = reader.CommandLine(opts, args)
    out_path =  'out'.join(config.filepath.rsplit('data', 1))
    models_generator = Models_Generator(config)
    models_generator.generate()
