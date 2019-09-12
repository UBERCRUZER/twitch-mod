# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:02:54 2018

@author: UBERCRUZER
"""

import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import os
import numpy as np

import spacy  # For preprocessing
from gensim.models.phrases import Phrases, Phraser

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

#file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'cass chat logs')
#
#
#corpus = []
#
#for filename in os.listdir(file_dir):
#    inputfile = os.path.join(file_dir, filename)
#    for line in open(inputfile, encoding="utf8"):
#        line = line.rstrip('\n') 
#        if not line.startswith('*'):
#            corpus.append(line[line.find('>')+2:])

#corpus



file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS664')
inputfile = os.path.join(file_dir, 'train_E6oV3lV.csv')
df = pd.read_csv(inputfile)

corpus = df['tweet']

df = []
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_non_alphanum, strip_numeric, strip_multiple_whitespaces, stem
for msg in corpus:
    string = remove_stopwords(msg)
    string = strip_punctuation(string)
    string = strip_non_alphanum(string)
    string = strip_numeric(string)
    string = strip_multiple_whitespaces(string)
    string = stem(string)
    df.append(string)

corpus = df

#out = pd.DataFrame(data=corpus)
#out.to_csv('chatOut.csv', index_label=False)

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]

max_epochs = 50
vec_size = 50
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

#save model
file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS664')
save_path = os.path.join(file_dir, 'd2v_twitter.model')
model.save(save_path)

##model = Doc2Vec.load(save_path)
##to find the vector of a document which is not in training data
#test_data = word_tokenize("you're a fucking racist".lower())
#v1 = model.infer_vector(test_data)
#
## to find most similar doc using tags
#similar_doc = model.docvecs.most_similar([v1], topn = 5)
#corpus[int(similar_doc[0][0])]

## to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
#print(model.docvecs['1'])
