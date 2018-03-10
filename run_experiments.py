"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) (directory, debate identifier or topic)
OPTIONS:
    --help        ,  -h: print this help message
    --classifier     -c: What classifier to use
    --image          -i: Where to export graphs
    --setup       ,  -s: Experiment setup to use
    --createdebate,  -d: CreateDebate post directory
    --topic       ,  -t: topic/seen target
    --4forums     ,  -f: 4Forums.com post directory
    --topic2      ,  -u: unseen target
------------------------------------------------------------
"""

import numpy as np
import reader, writer, preprocess, copy, tabulate, argparse
from twokenize_wrapper.twokenize import tokenize
from sklearn import svm
from sklearn.preprocessing import scale

STANCES = {'AGAINST': -1, 'NONE': 0, 'FAVOR': 1}

class Model_Wrapper:
    def __init__(self, model):
        self.model = model                                                                                                                                                                                                                                                                                      
    
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
        return list(map(lambda p: STANCES[p.label], debate.post_list))
    
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
        super(Experiment1, self).__init__(classifier, img_path, dir_cd)
        self.topic = topic
    
    def evaluate(self, rdr):
        mets = {}
        accs = []
        max_props = []
        for prefix in reader.subsetAZ(self.dir_cd + self.topic):
            print("Generating model")
            train_data = rdr.load_cd(topic, prefix, True)
            test_data = rdr.load_cd(topic, prefix)
            wvm_gen = writer.Writer_X1(train_data, test_data, self.topic, prefix)
            wvmodel = Model_Wrapper(wvm_gen.skipgram(15,4,151))
            train_arrays = wvmodel.vectorise_debate(train_data)
            test_arrays  = wvmodel.vectorise_debate(test_data)
            train_labels = wvmodel.stance_to_int(train_data)
            test_labels = wvmodel.stance_to_int(test_data)
            print("Training on", topic, "debates except for", prefix)
            self.classifier.fit(train_arrays,train_labels)
            print("Testing on", topic, "debate", prefix)
            
            # Evaluation Metrics
            accuracy = self.classifier.score(test_arrays, test_labels)
            accs += [accuracy]
            print("Accuracy:", accuracy)
            predicted = list(self.classifier.predict(test_arrays))
            metrics={'Metric': ['Precision', 'Recall', 'F-measure', 'Proportion in Training Data']}
            
            proportions=[]
            for stance, n in STANCES.items():
                #Avoid division by zero if testing data doesn't have the stance
                numerator = correctness(test_labels,predicted, n)
                denominator1 = predicted.count(n)
                denominator2 = test_labels.count(n)
                precision = numerator/denominator1 if denominator1 > 0 else "n/a"
                recall = numerator/denominator2 if denominator2 > 0 else "n/a"
                f_measure = "n/a"
                if denominator1 > 0 and denominator2 > 0:
                    f_measure = 2*precision*recall
                    if precision+recall > 0:
                        f_measure = f_measure/(precision+recall)
                proportion = train_labels.count(n)/len(train_labels)
                proportions+=[proportion]
                metrics[stance] = [precision, recall, f_measure, proportion]
            max_props += [max(proportions)]
            mets[prefix] = metrics
        return mets, accs, max_props
    
class Experiment2(Experiment):
    def __init__(self, classifier, img_path, dir_cd, dir_4f, seen_target, unseen_target):
        """
        A subclass of Experiment specifically for experiment setup 2.
        
        :param dir_4f       : the directory of the 4Forum.com files.
        :param seen_target  : the topic of the training data.
        :param unseen_target: the topic of the testing data.
        """
        super(Experiment2, self).__init__(classifier, img_path, dir_cd)
        self.dir_4f = dir_4f
        self.seen_target = seen_target
        self.unseen_target = unseen_target
    
    def evaluate(self, rdr):
        train_data = rdr.load_cd(self.seen_target,'ALL')
        rdm_dbt = rdr.load_4f(unseen_target)
        test_data = preprocess.Debate(unseen_target, [rdm_dbt])
        wvm_gen = writer.Writer(train_data, test_data)
        wvmodel = Model_Wrapper(wvm_gen.skipgram(15,4,151))
        train_arrays = wvmodel.vectorise_debate(train_data)
        test_arrays  = wvmodel.vectorise_debate(test_data)
        train_labels = wvmodel.stance_to_int(train_data)
        test_labels = wvmodel.stance_to_int(test_data)
        print("Training on all", self.seen_target, "debates")
        self.classifier.fit(train_arrays,train_labels)
        print("Testing on", unseen_target, "debate", rdm_dbt.post_id)
        print("Accuracy:", self.classifier.score(test_arrays, test_labels))

def correctness(actual, predicted, class_n):
    return sum([(x[0]==x[1] and x[0]==class_n) for x in zip(actual, predicted)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments setup 1 or 2.')
    parser.add_argument('--classifier', '-c', nargs='?', default ='', help='The classifier to use.')
    parser.add_argument('--image', '-i', nargs='?', default ='', help='Where to output graphs.')
    parser.add_argument('--setup', '-s', nargs='?', default = 0, type=int, help='Experiment setup to use.')
    parser.add_argument('--createdebate', '-d', nargs='?', default ='', help='The CreateDebate post directory.')
    parser.add_argument('--topic', '-t', nargs='?', default ='', help='The topic (setup 1) or seen target (setup 2)')
    parser.add_argument('--4forums', '-4', nargs='?', default ='', help='The 4Forums.com post directory.')
    parser.add_argument('--unseen', '-u', nargs='?', default ='', help='The unseen target (setup 2).')
    args = vars(parser.parse_args())
    
    classifiers = [svm.LinearSVC()] + list(map(lambda k: svm.SVC(kernel=k), ['linear', 'poly', 'sigmoid', 'rbf'])) #Will implement LSTMs
    classifier_names = ['liblinear', 'linear', 'Polynomial','Sigmoid','Radial Basis Function']
    classifier_dict = dict(zip(classifier_names, classifiers))
    valid_classifier = args['classifier'].lower() in classifier_names
    classifier_choice = args['classifier'].lower() if valid_classifier else reader.select_opt(classifier_names, "Select a classifier:")

    dir_cd = input("Where are the posts from CreateDebate stored?") if not args['createdebate'] else args['createdebate'] #./data/CreateDebate/
    rdr=reader.Reader(dir_cd)
    
    valid_setup = 0 < args['setup'] < 3
    setup_choice = 'setup ' + str(args['setup']) if valid_setup else reader.select_opt(['setup 1','setup 2'],'Select an experiment setup:')
    img_path = input("Where should graphs be exported to?") if not args['image'] else args['image']
    topic = rdr.select_topic() if args['topic'] not in rdr.dir_lst else args['topic']
    
    if setup_choice == 'setup 1':
        #Experiment 1 args: -i ./img/experiment_1/ -s 1 -d ./data/CreateDebate/
        classifier1 = copy.deepcopy(classifier_dict[classifier_choice]) 
        experiment1 = Experiment1(classifier1,img_path,dir_cd,topic)
        mets, accs, max_props = experiment1.evaluate(rdr)
        print("Mean Accuracy:", np.mean(accs))
        print("# of times accuracy was greater than proportion of most frequent stance in training data:", sum([x[0]>x[1] for x in zip(accs,max_props)]))
        for prefix, metrics in mets.items():
            print(prefix)
            print(tabulate.tabulate(metrics, headers="keys"))
        
    else:
        #Experiment 2 args: -i ./img/experiment_2/ -s 2 -d ./data/CreateDebate/ -4 ./data/fourforums/
        dir_4f = input("Where are the posts from 4Forums.com stored?") if not args['4forums'] else args['4forums'] #./data/fourforums/
        rdr.dir_4f = dir_4f
        unseen_target = rdr.select_target() if args['unseen'] not in rdr.topic_4f else args['unseen']
        classifier2 = copy.deepcopy(classifier_dict[classifier_choice]) 
        experiment2 = Experiment2(classifier2,img_path,dir_cd,dir_4f,topic,unseen_target)
        experiment2.evaluate(rdr)
