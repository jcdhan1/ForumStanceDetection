# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:56 2017

@author: aca15jch
"""
import os, sys, nltk
if os.path.abspath('./') not in sys.path:
    sys.path.append(os.path.abspath('./'))
if list(filter(lambda p: 'twokenize_wrapper' in p, sys.path))==[]:
    sys.path.append(os.path.abspath('./twokenize_wrapper'))
if list(filter(lambda p: 'readwrite' in p, sys.path))==[]:
    sys.path.append(os.path.abspath('./readwrite'))
print(sys.path)
nltk.download()
