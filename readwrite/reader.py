# -*- coding: utf-8 -*-
"""


@author: aca15jch
"""
import os

class CommandLine:
    def __init__(self, opts, args):
        p = '-p' in opts
        t = '-T' in opts
        if len(args)==2:
             self.filepath = args[0]
             self.topic = args[1]
        if len(args)==1:
            if t:
                self.filepath = args[0]
            if p:
                self.topic = args[0]
        #Flags supersede arguments if present
        if p:
            self.filepath = opts['-p']
        if t:
            self.topic = opts['-T']
            

def filefilter(f="", extension='.data', prefix='all', exclude=False):
    """
    Filter files by prefix or extension
    :param f        : the file name
    :param extension: the extension
    :param prefix   : the prefix
    :param exclude  : if true, do not return true for files with the prefix
    :return: True if it matches the conditions
    """
    return f.endswith(extension) and (prefix=='all' or (f.startswith(prefix) != exclude))

class Post:
    def __init__(self, body="", label="NONE", post_id=None, target=-1):
        """
        Constructor for Post object.
        :param body: the post's text body.
        :param label: takes values of "AGAINST", "NONE", or "FAVOR"
        :param post_id: a number uniquely identifying the post.
        :param target: Either the post_id of another post that this post responds to or the the number -1.
        """
        self.body = body
        self.label = label
        self.post_id = post_id
        self.target = target
    
    def __str__(self):
        """
        Overriden str method
        :return: The post's attributes as a string.
        """
        return '\n'.join(["Label  : " + self.label, "Post ID: " + str(self.post_id), "Target : " +  str(self.target), "Body   :\n" +  self.body])

class Debate:
    def __init__(self, posts_directory="", prefix='all', topic="", exclude=False):
        """
        Constructor for Debate object.
        :param posts_directory: Where the .data and .meta files are located.
        :param prefix: takes values of "all" or the majuscule Latin alphabet (formatted for CreateDebate files).
        :param exclude: whether to exclude or include files with said prefix.
        :param topic: the topic of the debate.
        """
        self.posts_directory = posts_directory
        self.prefix = prefix
        self.topic = topic
        dataList = list(map(lambda d_file: os.path.join(posts_directory, d_file), filter(lambda d_file:  filefilter(f=d_file, prefix=prefix, exclude=exclude), os.listdir(posts_directory))))
        dataList.sort()
        metaList = list(map(lambda m_file: os.path.join(posts_directory, m_file), filter(lambda m_file:  filefilter(m_file, '.meta', prefix, exclude), os.listdir(posts_directory))))
        metaList.sort()
        self.post_list=[]
        for x in range(0, len(dataList)):
            new_post = Post()
            with open(dataList[x], encoding="utf8") as fd:
                new_post.body = fd.read()
            with open(metaList[x], encoding="utf8") as fd:
                new_post.post_id=os.path.basename(dataList[x])[:1] + str(int(fd.readline().split("ID=",1)[1]))
                raw_target=int(fd.readline().split("PID=",1)[1])
                new_post.target = topic if raw_target==-1 else os.path.basename(dataList[x])[:1] + str(raw_target)
                raw_stance=fd.readline().split("Stance=",1)[1]
                if any(char.isdigit() for char in raw_stance):
                    if int(raw_stance) < 0:
                        new_post.label = "AGAINST"
                    elif int(raw_stance) > 0:
                        new_post.label = "FAVOR"
            self.post_list.append(new_post)
        self.post_list.sort(key=lambda p: p.post_id)
    
    def get_bodies(self):
        return list(map(lambda p: p.body, self.post_list))
    
    def get_targets(self):
        return list(map(lambda p: p.target, self.post_list))
    
    def get_labels(self):
        return list(map(lambda p: p.label, self.post_list))
    
    def get_post_ids(self):
        return list(map(lambda p: p.post_id, self.post_list))

class DebateClass1(Debate):
    def __init__(self, topic="", posts_directory="", prefix='all', exclude=False):
        """
        Extended to be specific to CreateDebate files
        :param posts_directory: Where the .data and .meta files are located.
        :param prefix: takes values of "all" or the majuscule Latin alphabet (formatted for CreateDebate files).
        :param exclude: whether to exclude or include files with said prefix.
        :param topic: the topic of the debate.
        """
        self.posts_directory = posts_directory
        self.prefix = prefix
        self.topic = topic
        dataList = list(map(lambda d_file: os.path.join(posts_directory, d_file), filter(lambda d_file:  filefilter(f=d_file, prefix=prefix, exclude=exclude), os.listdir(posts_directory))))
        dataList.sort()
        metaList = list(map(lambda m_file: os.path.join(posts_directory, m_file), filter(lambda m_file:  filefilter(m_file, '.meta', prefix, exclude), os.listdir(posts_directory))))
        metaList.sort()
        self.post_list=[]
        for x in range(0, len(dataList)):
            new_post = Post()
            with open(dataList[x], encoding="utf8") as fd:
                new_post.body = fd.read()
            with open(metaList[x], encoding="utf8") as fd:
                new_post.post_id=os.path.basename(dataList[x])[:1] + str(int(fd.readline().split("ID=",1)[1]))
                raw_target=int(fd.readline().split("PID=",1)[1])
                new_post.target = topic if raw_target==-1 else os.path.basename(dataList[x])[:1] + str(raw_target)
                raw_stance=fd.readline().split("Stance=",1)[1]
                if any(char.isdigit() for char in raw_stance):
                    if int(raw_stance) < 0:
                        new_post.label = "AGAINST"
                    elif int(raw_stance) > 0:
                        new_post.label = "FAVOR"
            self.post_list.append(new_post)
        self.post_list.sort(key=lambda p: p.post_id)
    
    def __str__(self):
        """
        Overriden str method
        :return: The debates's attributes as a string.
        """
        return '\n'.join(["Directory: " + str(self.posts_directory), "Prefix   : " + self.prefix, "Topic    : " + str(self.topic)])