#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:35:30 2017

@author: Chenhao
"""
#from __future__ import print_function
#from nltk.stem import *
#from nltk.stem.porter import *
#from nltk.stem.snowball import SnowballStemmer
#
#from nltk.corpus import stopwords
#from nltk.tokenize import wordpunct_tokenize

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso
from sklearn import metrics
import re


tweetText = []
tweetLabel = []

     

def reading_data_save_to_array(labelfilename,textfilename,train_percent):
    
    with open(labelfilename, 'r') as f:
        for line in f:
            line = line.strip()
            tweetLabel.append(line)
            
    
    with open(textfilename, 'r') as f:
        for line in f:
            line = line.strip()
            tweetText.append(line)
    
    vectorizer = CountVectorizer(stop_words='english')
    dictionary = np.column_stack((tweetLabel,tweetText))    
    
    np.random.seed(1)
    np.random.shuffle(dictionary)
   
    m = dictionary.shape[0]
    train_end = int(train_percent * m)
    train = dictionary[:train_end]
    train_x = train[:, 1:]
    train_y = train[:, 0]
    test = dictionary[train_end:]
    test_x = test[:, 1:]
    test_y  = test[:, 0]
    print (dictionary[:1])
    X_train_counts =  vectorizer.fit_transform(train_x[:,0])
    tfidf_transformer = TfidfTransformer( )
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    print(X_train_counts.toarray())
    clf = LogisticRegression()
#    print((vectorizer.get_feature_names()))
    clf = clf.fit(X_train_tfidf,train_y)

    X_new_counts = vectorizer.transform(test_x[:,0])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    print(X_new_tfidf.shape)
    predicted = clf.predict(X_new_tfidf)
    
    print (np.mean(predicted == test_y)  )
    print(metrics.classification_report(test_y, predicted))
    return train_x,train_y,test_x,test_y


    
    
    

    
print ("SNOWBALL")
train_x,train_y,test_x,test_y = reading_data_save_to_array('tweet_labels.labels','tweet_text_STEM.text',0.8)

print ("stop")
train_x,train_y,test_x,test_y = reading_data_save_to_array('tweet_labels.labels','tweet_text_STOP.text',0.8)


print ("BOTH")
train_x,train_y,test_x,test_y = reading_data_save_to_array('tweet_labels.labels','tweet_text_BOTH.text',0.8)

print ("Porter")
train_x,train_y,test_x,test_y = reading_data_save_to_array('tweet_labels.labels','tweet_text_POTER.text',0.8)

print ("PorterBOTH")
train_x,train_y,test_x,test_y = reading_data_save_to_array('tweet_labels.labels','tweet_text_PORTER_BOTH.text',0.8)

