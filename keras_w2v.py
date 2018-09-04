#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import string
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from get_embedding import sent_embedding
from keras.layers import Dropout

df = pd.read_csv('train_2kmZucJ.csv')

df['tweet']= df['tweet'].apply(lambda x: x.translate(None, string.punctuation))
df['tweet']= df['tweet'].apply(lambda x: x.translate(None, string.digits))

sentences = df['tweet'].tolist()
y = df['label'].tolist()
y  = pd.get_dummies(y)

X = []
for sentence in sentences:
    sent_emb = sent_embedding(sentence)
    X.append(sent_emb)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2,stratify=y)

model = Sequential()

model.add(Dense(5,input_dim=50,activation='relu'))
model.add(Dense(2,activation='sigmoid'))

model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
model.fit(np.array(X),np.array(y),nb_epoch=100,validation_data=(np.array(X_test),np.array(y_test)))

model.save('keras_w2v_300_final.h5')

print(model.summary())

# model.add(Dense(50,input_dim=200,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(2,activation='sigmoid'))
