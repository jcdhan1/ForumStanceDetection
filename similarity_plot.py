# -*- coding: utf-8 -*-
"""
Created on Tue May  1 07:07:06 2018

@author: aca15jch
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
line = plt.figure()

accs_linear_svm    = np.array([0.5241460541813898 , 0.5150300601202404 , 0.5439560439560439])
accs_conditional   = np.array([0.50412249705535928, 0.48897795594168569, 0.4945054945054945])
accs_bidirectional = np.array([0.5188457008244994 , 0.5030060120240480 , 0.5439560439560439])


sim_vals = np.array([0.77796435, 0.8471235, 0.85121942])
lsvm = plt.scatter(sim_vals,accs_linear_svm,color='r')
cond = plt.scatter(sim_vals,accs_conditional,color='g', marker='v')
bdrt = plt.scatter(sim_vals,accs_bidirectional,color='b', marker='^')
plt.legend((lsvm, cond, bdrt),
           ('Linear SVM', 'Conditional Encoding', 'Bidirectional Encoding'))

plt.show
plt.savefig('./results/similarityperformance.svg', format='svg')
plt.show()
