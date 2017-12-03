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
import os, sys, getopt, tabulate
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
from readwrite import reader
from gensim.models import Doc2Vec, Word2Vec
from sklearn import svm
from sklearn.preprocessing import scale

class Preprocessor:
    def __init__(self, train_debates, test_debate, prefix):
        self.train_debates = train_debates
        self.test_debate = test_debate
        self.prefix = prefix
        self.out_path = 'out'.join(config.filepath.rsplit('data', 1))
        self.model_dm = Doc2Vec.load(self.out_path + '/' + self.prefix + '.dmv')
        self.model_sg = Word2Vec.load(self.out_path + '/' + self.prefix + '.sgv')
        self.num_features = 151
        self.stance_dict = {'AGAINST': -1, 'NONE': 0, 'FAVOR': 1}
        
    #Distributed Memory Model
    def distributed_memory(self):
        train_against = list(filter(lambda p: p.label=="AGAINST", self.train_debates.post_list))
        train_none = list(filter(lambda p: p.label=="NONE", self.train_debates.post_list))
        train_favor = list(filter(lambda p: p.label=="FAVOR", self.train_debates.post_list))
        
        test_against = list(filter(lambda p: p.label=="AGAINST", self.test_debate.post_list))
        test_none = list(filter(lambda p: p.label=="NONE", self.test_debate.post_list))
        test_favor = list(filter(lambda p: p.label=="FAVOR", self.test_debate.post_list))
        
        train_rows = len(train_against) + len(train_none) + len(train_favor)
        train_arrays = np.zeros((train_rows, self.num_features))
        train_labels = np.zeros(train_rows)
        for i in range(len(train_against)):
            id_in_model = 'TRAIN_AGAINST_' + train_against[i].post_id
            train_arrays[i] = self.model_dm.docvecs[id_in_model]
            train_labels[i] = -1
        for j in range(len(train_against),len(train_against)+len(train_none)):
            id_in_model = 'TRAIN_NONE_' + train_none[j-len(train_against)].post_id
            train_arrays[j] = self.model_dm.docvecs[id_in_model]
            train_labels[j] = 0
        for k in range(len(train_against)+len(train_none),train_rows):
            id_in_model = 'TRAIN_FAVOR_' + train_favor[k-(len(train_against)+len(train_none))].post_id
            train_arrays[k] = self.model_dm.docvecs[id_in_model]
            train_labels[k] = 1
        test_rows = len(test_against) + len(test_none) + len(test_favor)
        test_arrays = np.zeros((test_rows, self.num_features))
        test_labels = np.zeros(test_rows)
        for i in range(len(test_against)):
            id_in_model = 'TEST_AGAINST_' + test_against[i].post_id
            test_arrays[i] = self.model_dm.docvecs[id_in_model]
            test_labels[i] = -1
        for j in range(len(test_against),len(test_against)+len(test_none)):
            id_in_model = 'TEST_NONE_' + test_none[j-len(test_against)].post_id
            test_arrays[j] = self.model_dm.docvecs[id_in_model]
            test_labels[j] = 0
        for k in range(len(test_against)+len(test_none),test_rows):
            id_in_model = 'TEST_FAVOR_' + test_favor[k-(len(test_against)+len(test_none))].post_id
            test_arrays[k] = self.model_dm.docvecs[id_in_model]
            test_labels[k] = 1
        return train_arrays, train_labels, test_arrays, test_labels

    #Skip-Gram Model
    def vectorise(self, body):
        v_of_vectors = np.zeros(self.num_features).reshape((1, self.num_features))
        words = 0
        for w in body:
            try:
                v_of_vectors += self.model_sg[w].reshape((1, self.num_features))
                words += 1
            except KeyError:
                continue
        if words>0:
            v_of_vectors /= words
        return v_of_vectors
    
    def skipgram(self):
        train_arrays = scale(np.concatenate(list(map(lambda b: self.vectorise(b), self.train_debates.get_bodies()))))
        test_arrays = scale(np.concatenate(list(map(lambda b: self.vectorise(b), self.test_debate.get_bodies()))))
        train_labels = list(map(lambda l: self.stance_dict[l], self.train_debates.get_labels()))
        test_labels = list(map(lambda l: self.stance_dict[l], self.test_debate.get_labels()))
        return train_arrays, train_labels, test_arrays, test_labels
    


def multiscore(train_arrays, train_labels, test_arrays, test_labels):
    svms = [svm.LinearSVC()] + list(map(lambda k: svm.SVC(kernel=k), ['linear', 'poly', 'sigmoid', 'rbf']))
    scores = []
    for c in svms:
        c.fit(train_arrays, train_labels)
        scores.append(c.score(test_arrays, test_labels)) 
    return scores

def tuplist(d):
    return [(k, v) for k, v in d.items()]

def graph_data(config, scores_dict, bar_width):
    n_groups = len(scores_dict)
    svm_type = ['Liblinear', 'Linear', 'Polynomial','Sigmoid','Radial Basis Function']
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    for n in range(5):
        path_split = config.filepath.rsplit('data', 1)
        img_path = 'img'.join(path_split)
        dm_scores = list(map(lambda entry: entry[1][1][n], tuplist(scores_dict)))
        sg_scores = list(map(lambda entry: entry[1][2][n], tuplist(scores_dict))) 
        dm_mu1, dm_sd1 = tuple(map(lambda fn: fn(dm_scores), (np.mean, np.std)))
        sg_mu1, sg_sd1 = tuple(map(lambda fn: fn(sg_scores), (np.mean, np.std)))
        #Copy each of the scores by how big the debate is to get weighted means and standard deviations
        dm_scores_expanded = np.concatenate(list(map(lambda s: s[1][0]*[s[1][1][n]], tuplist(scores_dict))))
        dm_mu2, dm_sd2 = tuple(map(lambda fn: fn(dm_scores_expanded), (np.mean, np.std)))
        sg_scores_expanded = np.concatenate(list(map(lambda s: s[1][0]*[s[1][2][n]], tuplist(scores_dict))))
        sg_mu2, sg_sd2 = tuple(map(lambda fn: fn(sg_scores_expanded), (np.mean, np.std)))
        print('\\begin{table}[ht]\n\\centering\n')
        print(tabulate.tabulate([['','Mean mu','Standard Deviation sigma', 'Scaled mu', 'Scaled sigma'],['D.M.',dm_mu1,dm_sd1,dm_mu2,dm_sd2],['S.G.',sg_mu1,sg_sd1,sg_mu2,sg_sd2]], [svm_type[n],'','','',''], tablefmt='latex_booktabs').replace('mu', '$\mu$').replace('sigma', '$\sigma$').replace('lll', 'rll').replace('\\toprule', '').replace('\\bottomrule',''))
        print('\\end{table}\n')
        g_type = tuple(map(lambda g: g + svm_type[n].replace(" ", "") + '.png', ['unscaled', 'scaled', 'histogram']))
        unscaled, scaled, histogram = g_type
        print(tabulate.tabulate([map(lambda g: 'includegraphics img' + path_split[1] + '/' + g, g_type)], ['Scores', 'Scaled Scores', 'Distribution of Scores'], tablefmt='latex_booktabs').replace('png','png}').replace('includegraphics ','\includegraphics[width=0.3\\textwidth]{'))
        
        #Scores
        rects1 = plt.bar(index, dm_scores, bar_width, color='r', label='Distributed Memory')
        rects2 = plt.bar(index + bar_width, sg_scores, bar_width, color='c', label='Skip-Gram')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,1))
        plt.xlabel('Debate')
        plt.ylabel('Scores')
        plt.title('Scores by Debate')
        plt.xticks(index+0.5*bar_width, list(map(lambda entry: entry[0], tuplist(scores_dict))))
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path + '/' + unscaled, bbox_inches='tight')
        plt.show()
        
        #Scaled
        dm_scores_scaled = list(map(lambda entry: 15*entry[1][0]*entry[1][1][n]/len(dm_scores_expanded), tuplist(scores_dict)))
        sg_scores_scaled = list(map(lambda entry: 15*entry[1][0]*entry[1][2][n]/len(dm_scores_expanded), tuplist(scores_dict)))
        rects1 = plt.bar(index, dm_scores_scaled, bar_width, color='r', label='Distributed Memory')
        rects2 = plt.bar(index + bar_width, sg_scores_scaled, bar_width, color='c', label='Skip-Gram')     
        plt.axis((x1,x2,0,4))
        plt.xlabel('Debate')
        plt.ylabel('Scaled scores')
        plt.title('Scaled Scores by Debate')
        plt.xticks(index+0.5*bar_width, list(map(lambda entry: entry[0], tuplist(scores_dict))))
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path + '/' + scaled, bbox_inches='tight')
        plt.show()
        
        #Histogram
        bins=np.linspace(0, 1, 50)
        plt.hist(dm_scores, color="r", bins=bins, alpha=0.5, label='Distributed Memory')
        plt.hist(sg_scores, color="c", bins=bins, alpha=0.5, label='Skip-Gram')   
        plt.xlabel('Score')
        plt.title('Distribution of Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path + '/' + histogram, bbox_inches='tight')
        plt.show()
        print('\\newline\n\\newline')

if __name__ == '__main__':
    #Example: ../data/CreateDebate/obama Obama
    
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

    """Although printing the set will show the letters sorted, the letters are not accessed in order when
    #the set's converted to a list or used in a for-loop hence it must be converted to a list and then sorted."""
    subsetAZ = list(set(map(lambda f: f[0], os.listdir(config.filepath))))
    subsetAZ.sort()
    
    scores_dict = dict.fromkeys(subsetAZ)
    
    print('     Liblinear     |Linear        |Polynomial    |Sigmoid       |Radial Basis Function')
    for prefix in subsetAZ:
        #Training data: files about config.topic that do not begin with the prefix
        train_debates = reader.Debate(config.filepath, prefix, config.topic, True)
        #Testing data: files about config.topic that begin with the prefix
        test_debate = reader.Debate(config.filepath, prefix, config.topic)
        
        #Preparing Vectors for Classifier
        
        #Score for multiple SVMs
        prepro=Preprocessor(train_debates, test_debate, prefix)
        dm_scores = multiscore(*prepro.distributed_memory())
        print(prefix, 'DM', "|".join(map(lambda sc: str.format('{0:.12f}',sc), dm_scores)),len(test_debate.post_list))
        sg_scores = multiscore(*prepro.skipgram())
        print('  SG',"|".join(map(lambda sc: str.format('{0:.12f}',sc), sg_scores)),len(test_debate.post_list))
        
        scores_dict[prefix] = (len(test_debate.post_list), dm_scores, sg_scores)
    print('\\subsubsection*{Topic: ' + config.topic + '}')
    graph_data(config, scores_dict=scores_dict, bar_width=0.5)
