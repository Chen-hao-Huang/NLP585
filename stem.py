#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:35:30 2017

@author: Chenhao
"""
from __future__ import print_function
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso
from sklearn import metrics
import re


tweetText = []
tweetLabel = []

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 


def stop(plurals):
    
    pl = re.sub("[^\w]", " ",  plurals).split()
    singles = []
    for word in pl:
        if word.lower() not in stop_words:
            singles.append(word)
    return ' '.join(singles)

def porter(plurals):
    stemmer = PorterStemmer()
    pl = re.sub("[^\w]", " ",  plurals).split()
    singles = [stemmer.stem(plural) for plural in pl]

    return ' '.join(singles)
     
def stem(plurals):
    stemmer = SnowballStemmer("english")
    pl = re.sub("[^\w]", " ",  plurals).split()
    singles = [stemmer.stem(plural) for plural in pl]

    return ' '.join(singles)

def STEM(textfilename,out):
    f2  =  open(out, 'w') 
    with open(textfilename, 'r') as f:
        for line in f:
            line = line.strip()
            line = stem(line)
            print(line)
            f2.write(line+ '\n')
            tweetLabel.append(line)
            
def PORTER(textfilename,out):
    f2  =  open(out, 'w') 
    with open(textfilename, 'r') as f:
        for line in f:
            line = line.strip()
            line = porter(line)
            print(line)
            f2.write(line+ '\n')
            tweetLabel.append(line)

def STOP(textfilename,out):
    f2  =  open(out, 'w') 
    with open(textfilename, 'r') as f:
        for line in f:
            line = line.strip()
            line = stop(line)
            print(line)
            f2.write(line+ '\n')
            tweetLabel.append(line)
        
            
def BOTH(textfilename,out):
    f2  =  open(out, 'w') 
    with open(textfilename, 'r') as f:
        for line in f:
            line = line.strip()
            line = stem(line)
            line = stop(line)
            print(line)
            f2.write(line+ '\n')
            tweetLabel.append(line) 
            
                
def PORTER_BOTH(textfilename,out):
    f2  =  open(out, 'w') 
    with open(textfilename, 'r') as f:
        for line in f:
            line = line.strip()
            line = porter(line)
            line = stop(line)
            print(line)
            f2.write(line+ '\n')
            tweetLabel.append(line)    
            
#STEM('tweet_text.text','tweet_text_STEM.text')
#STOP('tweet_text.text','tweet_text_STOP.text')
#BOTH('tweet_text.text','tweet_text_BOTH.text')
PORTER('tweet_text.text','tweet_text_PORTER.text')
PORTER_BOTH('tweet_text.text','tweet_text_PORTER_BOTH.text')
