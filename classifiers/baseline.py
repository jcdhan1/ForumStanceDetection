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
    
class Multiclassifer:
    def __init__(self,train_arrays, train_labels, test_arrays, test_labels):
        self.train_arrays=train_arrays
        self.train_labels=train_labels
        self.test_arrays=test_arrays
        self.test_labels=test_labels
        self.classes=list(set(test_labels))
        self.svms = [svm.LinearSVC()] + list(map(lambda k: svm.SVC(kernel=k), ['linear', 'poly', 'sigmoid', 'rbf']))
        
        #Train every SVM on train_arrays
        for c in self.svms:
            c.fit(self.train_arrays, self.train_labels)
            
        #Test every SVM on test_arrays and store the evaluation metrics
        self.metrics=list(map(lambda c: self.eval_metrics(c),self.svms))

    #Quadruple of accuracy, a precision list, a recall list and an F1 score list.
    def eval_metrics(self,classifier):
        accu = classifier.score(self.test_arrays, self.test_labels) #accuracy
        pred = classifier.predict(self.test_arrays) #predictions
        corr=[] #numerator of precision and recall for each class
        classed = []
        actual = []
        for i in range(len(self.classes)):
            corr_count=0
            classed_count=0
            for j in range(len(self.test_labels)):
                if pred[j]==self.classes[i]:
                    classed_count +=1
                    if self.test_labels[j]==self.classes[i]:
                        corr_count += 1
            corr.append(corr_count) #How many are correctly classified as classes[i]
            classed.append(classed_count) #How many are classified as classes[i]
            actual.append(list(self.test_labels).count(self.classes[i])) #How many are actually in classes[i]
        
        
        prc = []
        
        print("correct:", corr)
        print("classed as:", classed)
        
        rcl = []
        f1  = []
        
    
        return (accu,prc,rcl,f1)
    
def tuplist(d):
    return [(k, v) for k, v in d.items()]

def graph_data(config, met_dict, bar_width):
    n_groups = len(met_dict)
    svm_type = ['Liblinear', 'Linear', 'Polynomial','Sigmoid','Radial Basis Function']
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    for n in range(5):
        path_split = config.filepath.rsplit('data', 1)
        img_path = 'img'.join(path_split)
        dm_acc = list(map(lambda entry: entry[1][1][0][n], tuplist(met_dict)))
        sg_acc = list(map(lambda entry: entry[1][2][0][n], tuplist(met_dict))) 
        dm_acc_mu, dm_acc_sd = tuple(map(lambda fn: fn(dm_acc), (np.mean, np.std)))
        sg_acc_mu, sg_acc_sd = tuple(map(lambda fn: fn(sg_acc), (np.mean, np.std)))
        print('\\begin{table}[ht]\n\\centering\n')
        print(tabulate.tabulate([['','Mean mu','Standard Deviation sigma'],['D.M.',dm_acc_mu,dm_acc_sd],['S.G.',sg_acc_mu,sg_acc_sd]], [svm_type[n],'',''], tablefmt='latex_booktabs').replace('mu', '$\mu$').replace('sigma', '$\sigma$').replace('lll', 'rll').replace('\\toprule', '').replace('\\bottomrule',''))
        print('\\end{table}\n')
        g_type = tuple(map(lambda g: g + svm_type[n].replace(" ", "") + '.png', ['acc_by_debate', 'acc_hist']))
        acc_by_debate, histogram = g_type
        print(tabulate.tabulate([map(lambda g: 'includegraphics img' + path_split[1] + '/' + g, g_type)], ['Accuracies', 'Distribution of Accuracies'], tablefmt='latex_booktabs').replace('png','png}').replace('includegraphics ','\includegraphics[width=0.3\\textwidth]{'))
        
        #Accuracies
        rects1 = plt.bar(index, dm_acc, bar_width, color='r', label='Distributed Memory')
        rects2 = plt.bar(index + bar_width, sg_acc, bar_width, color='c', label='Skip-Gram')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,1))
        plt.xlabel('Debate')
        plt.ylabel('Accuracies')
        plt.title('Accuracies by Debate')
        plt.xticks(index+0.5*bar_width, list(map(lambda entry: entry[0], tuplist(met_dict))))
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path + '/' + acc_by_debate, bbox_inches='tight')
        plt.show()
        
        #Histogram
        bins=np.linspace(0, 1, 50)
        plt.hist(dm_acc, color="r", bins=bins, alpha=0.5, label='Distributed Memory')
        plt.hist(sg_acc, color="c", bins=bins, alpha=0.5, label='Skip-Gram')   
        plt.xlabel('Score')
        plt.title('Distribution of Accuracies')
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
    
    met_dict = dict.fromkeys(subsetAZ)
    
    print('       Liblinear     |Linear        |Polynomial    |Sigmoid       |Radial Basis Function')
    for prefix in subsetAZ:
        #Training data: files about config.topic that do not begin with the prefix
        train_debates = reader.Debate(config.filepath, prefix, config.topic, True)
        #Testing data: files about config.topic that begin with the prefix
        test_debate = reader.Debate(config.filepath, prefix, config.topic)
        
        #Preparing vectors for classifiers
        prepro=Preprocessor(train_debates, test_debate, prefix)
        
        #Get metrics from multiple SVMs
        dm_classifiers = Multiclassifer(*prepro.distributed_memory())
        dm_met = list(zip(*dm_classifiers.metrics))
        print(prefix, 'DM  ', "|".join(map(lambda sc: str.format('{0:.12f}',sc), dm_met[0])),len(test_debate.post_list))
#        for i in range(1,len(dm_met)):
#            met_sym = ('prc','rcl',' F1')
#            for j in range(3):
#                print('  ','{: 00}'.format(+1),met_sym[j], "|".join(map(lambda sc: str.format('{0:.12f}',sc), dm_met[i][j])))
#        
        sg_classifiers = Multiclassifer(*prepro.skipgram())
        sg_met = list(zip(*sg_classifiers.metrics))
        print('  SG  ',"|".join(map(lambda sc: str.format('{0:.12f}',sc), sg_met[0])),len(test_debate.post_list))
        
        met_dict[prefix] = (len(test_debate.post_list), dm_met, sg_met)
    print('\\subsubsection*{Topic: ' + config.topic + '}')
    graph_data(config, met_dict=met_dict, bar_width=0.5)
