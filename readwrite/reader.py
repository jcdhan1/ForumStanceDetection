# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:55:12 2017

@author: aca15jch
"""
import preprocess
import os
import csv
import re
import json
import random

class Reader:
    def __init__(self, __dir_cd,__dir_4f=''):
        """
        Constructor for Reader.
        
        :param __dir_cd: the directory of the CreateDebate files.
        :param __dir_4f: the directory of the 4Forum.com files.
        """
        self.__dir_cd = __dir_cd
        self.__dir_4f = __dir_4f
        #Import 4Forum topic annotations as a dictionary and an inverted dictionary
        self.topic_dict = {}
        self.inv_topic_dict = {}
        if __dir_4f:
            with open(__dir_4f + 'annotations/topic.csv') as csvfile:
                self.topic_dict = dict(map(lambda dbt: (int(dbt[0]),re.sub(r"^\W+|\W+$", "", dbt[1])), list(csv.reader(csvfile, delimiter=','))[1:]))
            self.topic_4f = set(self.topic_dict.values())
            for tpc in self.topic_4f:
                file_names = []
                for k, v in self.topic_dict.items():
                    if v == tpc:
                        file_names.append(k)
                self.inv_topic_dict[tpc] = file_names
        #Valid CreateDebate topics
        self.dir_lst = os.listdir(self.__dir_cd)
    
    #The lists of topics must be updated if the path to each data-set are changed
    @property
    def dir_cd(self):
        return self.__dir_cd

    @dir_cd.setter
    def dir_cd(self, dir_cd):
        self.__dir_cd = dir_cd
        self.dir_lst = os.listdir(self.dir_cd)
            
    @property
    def dir_4f(self):
        return self.__dir_4f

    @dir_4f.setter
    def dir_4f(self, dir_4f):
        self.__dir_4f = dir_4f
        self.dir_lst = os.listdir(self.dir_cd)
        with open(self.__dir_4f + 'annotations/topic.csv') as csvfile:
            self.topic_dict = dict(map(lambda dbt: (int(dbt[0]),re.sub(r"^\W+|\W+$", "", dbt[1])), list(csv.reader(csvfile, delimiter=','))[1:]))
        self.topic_4f = set(self.topic_dict.values())
        for tpc in self.topic_4f:
            file_names = []
            for k, v in self.topic_dict.items():
                if v == tpc:
                    file_names.append(k)
            self.inv_topic_dict[tpc] = file_names
    
    def load_cd(self, topic_dir="", prefix='', exclude=False):
        """
        Load from CreateDebate dataset
        
        :param topic_dir: the directory of the debate to load.
        :param prefix   : the prefix of the debate to load.
        :param exclude  : load posts from debates excluding those represented by the prefix.
        :return         : A Debate object with a post_list consisting only of Post_CD objects.
        """
        tpc_dir = topic_dir
        if not tpc_dir:
            tpc_dir = self.select_topic()
        subset_az = ['ALL'] + subsetAZ(self.__dir_cd + tpc_dir)
        pfx = prefix
        if (not pfx) or (pfx not in subset_az):
            while (pfx not in subset_az):
                pfx = input("Select a debate from:\n" + str(subset_az) + "\n").upper()
        dataList = list(map(lambda d_file: os.path.join(self.__dir_cd + tpc_dir, d_file), filter(lambda d_f:  filefilter(f=d_f, prefix=pfx, exclude=exclude), os.listdir(self.__dir_cd + tpc_dir))))
        dataList.sort()
        metaList = list(map(lambda m_file: os.path.join(self.__dir_cd + tpc_dir, m_file), filter(lambda m_f:  filefilter(m_f, '.meta',pfx, exclude), os.listdir(self.__dir_cd + tpc_dir))))
        metaList.sort()
        post_list=[]
        for x in range(0, len(dataList)):
            pfx_actual = os.path.basename(dataList[x])[:1]
            new_post = preprocess.Post_CD(pfx_actual)
            with open(dataList[x], encoding="utf8") as fd:
                new_post.body = fd.read()
            with open(metaList[x], encoding="utf8") as fd:
                new_post.post_id=pfx_actual + str(int(fd.readline().split("ID=",1)[1]))
                raw_target=int(fd.readline().split("PID=",1)[1])
                new_post.topic = tpc_dir if raw_target==-1 else os.path.basename(dataList[x])[:1] + str(raw_target)
                raw_stance=fd.readline().split("Stance=",1)[1]
                if any(char.isdigit() for char in raw_stance):
                    if int(raw_stance) < 0:
                        new_post.label = "AGAINST"
                    elif int(raw_stance) >= 0:
                        new_post.label = "FAVOR"
            post_list.append(new_post)
        post_list.sort(key=lambda p: p.post_id)
        dbt = preprocess.Debate(tpc_dir, post_list)
        return dbt
        
    def load_4f(self, unseen_target = "", n=1):
        """
        Load random posts from the 4Forums.com dataset and prompt the user to classify them.
        
        :param unseen_target: The topic of the posts.
        :param n            : How many posts to select.
        :return             : A Post object.
        """
        selected_topic = self.select_target(unseen_target)
        post_list = []
        for i in range(0,n):
            while True:
                fn = random.choice(self.inv_topic_dict[selected_topic])
                raw_debate = json.load(open(self.__dir_4f+'discussions/'+str(fn)+'.json'))[:-2][0]
                raw_post = random.choice(raw_debate)
                post_id = str(fn) + '/' + str(raw_post[0])
                if post_id not in list(map(lambda p:p.post_id, post_list)):
                    break
            body = raw_post[3]
            #If a post_title exists (stored in a dictionary), it may contain an opinion, hence it is concatenated with the body
            maybe_dict = raw_post[4]
            if 'post_info' in maybe_dict:
                post_info = maybe_dict['post_info']
                if 'post_title' in post_info:
                    body = post_info['post_title'] + '\n' + body
            new_post = preprocess.Post(body, 'Choose a Stance', post_id, selected_topic)
            print(new_post)
            new_post.label = select_opt(["AGAINST","NONE", "FAVOR"],"What is the stance of this post?")
            post_list += [new_post]
        dbt = preprocess.Debate(unseen_target, post_list)
        return dbt
    
    def select_target(self, given_target=''):
        lst = list(self.topic_4f)
        if given_target not in lst:
            print("{:<5s} | {:<23s} | {:>12s}".format('Input','Topic','# of Debates'))
            return select_opt(lst,'Select a topic: ',self.inv_topic_dict)
        else:
            return given_target

    def select_topic(self):
        print('Input | Directory')
        topic = select_opt(self.dir_lst, 'Select a topic: ')
        return topic

def select_opt(opt,prompt,dct=""):
    """
    Prompts the user to select a value from a list.
    
    :param opt   : the options.
    :param prompt: The prompt string.
    :param dct   : display an optional third column
    :return      : What the user selected from opt.
    """
    for i in range(len(opt)):
        formatting=("{:<5d} | {:<23s}" + (" | {:>12d}" if dct else ""))
        print(formatting.format(i+1, opt[i], len(dct[opt[i]])) if dct else formatting.format(i+1, opt[i]))
    return opt[input_int(prompt,0,len(opt))-1]
    

def input_int(prompt, lower,upper):
    """
    Forces an input to be an integer between two values.
    
    :param prompt: The prompt string.
    :param lower : The lowest integer allowed.
    :param upper : The greatest integer allowed.
    :return      : An int between lower and greater.
    """
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if value < lower or value > upper:
            print("Sorry, your response is out of bounds.")
            continue
        else:
            break
    return value

def filefilter(f="", extension='.data', prefix='ALL', exclude=False):
    """
    Filter files by prefix or extension
    
    :param f        : the file name
    :param extension: the extension
    :param prefix   : the prefix
    :param exclude  : if true, do not return true for files with the prefix
    :return         : True if it matches the conditions
    """
    return f.endswith(extension) and (prefix=='ALL' or (f.startswith(prefix) != exclude))

def subsetAZ(filepath):
    """
    Although printing the set will show the letters sorted, the letters are not accessed in order when
    the set's converted to a list or used in a for-loop hence it must be converted to a list and then sorted.
    
    :param filepath: A subdirectory in one of the CreateDebate topic directories.
    :return        : The distinct letters files have.
    """
    subset_az = list(set(map(lambda f: f[0], os.listdir(filepath))))
    subset_az.sort()
    return subset_az

if __name__ == '__main__':
    #Instantiate a Reader
    rdr = Reader('../data/CreateDebate/', '../data/fourforums/')
    #Load a debate from the CreateDebate dataset that the user has been prompted to select.
    dbt1 = rdr.load_cd()
    
    #Load a random post from the 4Forums.com dataset and prompt the user to classify it.
    dbt2 = rdr.load_4f(n=3)
    
    print("Learn from these:")
    for p in dbt1.post_list:
        print(p)
    
    print("Classify these:")
    for p in dbt2.post_list:
        print(p)
