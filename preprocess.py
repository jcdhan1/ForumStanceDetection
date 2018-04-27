# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:18:33 2018

@author: aca15jch
"""

class Post:
    def __init__(self, body="", label="NONE", post_id=None, topic=""):
        """
        Constructor for Post.
        
        :param body: the post's text body.
        :param label: takes values of "AGAINST" or "FAVOR".
        :param post_id: a number uniquely identifying the post.
        :param topic: the topic of the post.
        """
        self.body = body
        self.label = label
        self.post_id = post_id
        self.topic = topic
        self.user = ""

    def __str__(self):
        """
        Overriden str method
        
        :return: The post's attributes as a string.
        """
        rtn0 = '\n'.join(["Label  : " + self.label, "Post ID: " + str(self.post_id)])
        if isinstance(self, Post_CD):
            rtn0 += "\nPrefix : " + self.prefix
        if self.user:
            rtn0 += "\nUser   : " + self.user
        return rtn0 + '\n' + '\n'.join(["Topic  : " +  str(self.topic), "Body   :\n" +  self.body])

class Post_CD(Post):
    def __init__(self, prefix, body="", label="NONE", post_id=None, topic=""):
        """
        Constructor for a subclass specific for CreateDebate
        
        :param prefix: The filename of the .data and .meta files it is sourced from.
        """
        super(Post_CD, self).__init__(body,label,post_id,topic)
        self.prefix = prefix
        
class Debate:
    def __init__(self, topic="", post_list=[]):
        """
        Constructor for Debate, a collection of posts with the same topic.
        
        :param topic: the topic of the debate.
        :param post_list: the posts in the debate.
        """
        self.topic = topic
        self.post_list=post_list
        self.post_list.sort(key=lambda p: p.post_id)

if __name__ == '__main__':
    #Instantiate a Post
    plain_post = Post("420", "FAVOR", 999, "Marijuana")
    
    #Instantiate a Post_CD
    post_cd = Post_CD('A',"blaze it", "FAVOR", 1000, "Marijuana")
    
    print(plain_post)
    print(post_cd)
