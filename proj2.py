# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:31:57 2018

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
#download if needed
#nltk.download('punkt')

##--------------------------- CHAT MODEL --------------------------------------
##import model
#file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS664')
#save_path = os.path.join(file_dir, 'd2v.model')
#model= Doc2Vec.load(save_path)

##read in training messages
#file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS664')
#inputfile = os.path.join(file_dir, 'chatOuttagged.csv')
#df = pd.read_csv(inputfile)
#df = df.iloc[:, 1:]
##-----------------------------------------------------------------------------


##-----------------------TWITTER MODEL-----------------------------------------

#import model
file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS664')
save_path = os.path.join(file_dir, 'd2v_twitter.model')
model= Doc2Vec.load(save_path)

#read in training messages
file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS664')
inputfile = os.path.join(file_dir, 'train_E6oV3lV.csv')
df = pd.read_csv(inputfile)


from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_non_alphanum, strip_numeric, strip_multiple_whitespaces, stem


messages = df.iloc[:,2]

temp = []

for msg in messages:
    string = remove_stopwords(msg)
    string = strip_punctuation(string)
    string = strip_non_alphanum(string)
    string = strip_numeric(string)
    string = strip_multiple_whitespaces(string)
    string = stem(string)

    temp.append(string)

df = pd.DataFrame({'tweet': temp, 'class': df.iloc[:,1]})

##-----------------------------------------------------------------------------


#df.iloc[:, -1].value_counts()

from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
#from keras.regularizers import l2
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from keras import optimizers


messages = df.iloc[:,0].tolist()

trainVec = []

for msg in messages:
    test_data = word_tokenize(msg.lower())
    trainVec.append(model.infer_vector(test_data))

xVec = np.array(trainVec)

y = np.array(df.iloc[:,-1].tolist())
y_encoded = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(xVec, y_encoded, test_size=0.10, 
                                                    random_state=0)

x_train.shape,y_train.shape,x_test.shape,y_test.shape

input_dim = xVec.shape[1]
output_dim = y_encoded.shape[1]
layer1 = 25
#layer2 = 25
epochs = 50

#build model
model = Sequential()

#Hidden Layer-1
model.add(Dense(layer1, 
                input_dim=input_dim, 
                activation='relu'))
model.add(Dropout(0.2, noise_shape=None, seed=None))


##Hidden Layer-2
#model.add(Dense(layer2, 
#                activation='relu'))
#model.add(Dropout(0.3, noise_shape=None, seed=None))


##----------------------FOR TWITCH---------------------------------------------
##Output layer
#model.add(Dense(output_dim, 
#                activation='softmax'))
##Compile model 
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', 
#              optimizer=sgd, 
#              )
##-----------------------------------------------------------------------------

##---------------------FOR TWITTER---------------------------------------------
#Compile model
#Output layer
model.add(Dense(1, 
                activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', 
              optimizer=sgd, 
              )


y = np.array(df.iloc[:,-1].tolist())

x_train, x_test, y_train, y_test = train_test_split(xVec, y, test_size=0.10, 
                                                    random_state=0)

##-----------------------------------------------------------------------------

model.summary()

model.fit(x_train, y_train, epochs=epochs)

preds = model.predict(x_test)

#np.argmax(preds, axis=1)
#np.argmax(y_test, axis=1)

#confusion_matrix(np.argmax(y_test, axis=1), np.argmax(preds, axis=1))

confusion = preds

threshold = 0.1202
y_pred = (preds > threshold)

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred))

#%%

#
##because i wanted to see how it fit on the training data.
#preds2 = model.predict(x_train)
#
#np.argmax(preds2, axis=1)
#np.argmax(y_train, axis=1)
#
##confusion_matrix(np.argmax(y_train, axis=1), np.argmax(preds2, axis=1))
#
#confusion_matrix(y_train,preds2)
#
#
#confusion = preds2
#
#threshold = 0.1
#ytrain_pred = (preds2 > threshold)
#
#print(confusion_matrix(y_train, ytrain_pred))
#print(f1_score(y_train, ytrain_pred))






    
    