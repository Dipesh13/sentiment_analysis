#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import string
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.model_selection import train_test_split
from get_embedding import sent_embedding

df = pd.read_csv('train_2kmZucJ.csv')
# Pre-Processing : remove punctations.
# digits lower case in sent_embediing.
# stop words left. use nltk for that
# decode left
df['tweet']= df['tweet'].apply(lambda x: x.translate(None, string.punctuation))
df['tweet']= df['tweet'].apply(lambda x: x.translate(None, string.digits))
# df['data']= df['data'].apply(lambda x: x.decode('utf-8'))
sentences = df['tweet'].tolist()
y = df['label'].tolist()
# y  = pd.get_dummies(y)

X = []
for sentence in sentences:
    sent_emb = sent_embedding(sentence)
    X.append(sent_emb)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2,stratify=y)

model = Sequential()
model.add(Dense(1,input_dim=50,activation='sigmoid'))

model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
model.fit(np.array(X_train),np.array(y_train),nb_epoch=100,validation_data=(np.array(X_test),np.array(y_test)))

model.save('keras_logreg.h5')

print(model.summary())

# load model
# clf = load_model('keras_operators.h5')

# print(clf.predict(np.array(X_test)))