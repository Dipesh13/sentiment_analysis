#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
np.random.seed(0)
import string
from sklearn.model_selection import train_test_split
from get_embedding import sent_embedding
from keras.models import Sequential , Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)

# data = open('dinos.txt', 'r').read()
# data = data.lower()

# df = pd.DataFrame({"data":["dogs fly high","football","call me now","see you tom","hey there","greek salad","indian football league","on my way","manchester united","hello"],"label":[1,1,0,0,1,1,0,0,1,0]})

df = pd.read_csv("train.csv")


char_corpus = []
X = df['data'].tolist()
y = df['label']

for sent in X:
    for ch in sent:
        char_corpus.append(ch)

X = [list(sent) for sent in X]


# for sent in X:
#     print(sent)

chars = list(set(char_corpus))
# print(chars)

length = [len(sent) for sent in X]
max_len = max(length)
print(max_len)

data_size = sum(length) #total number of characters in file
vocab_size = len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
# print(ix_to_char)


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def char_to_indices(X, char_to_ix, max_len):

    # m = X.shape[0]
    m = len(X)

    X_indices = np.zeros((m, max_len))

    for i in range(0,m):
        sentence = [w for w in X[i]]
        sent = list(sentence)
        # print(sent)
        j = 0
        for c in sent:
            # print(c)
            X_indices[i, j] = char_to_ix[c]
            j += 1

    return X_indices

# X1 = np.array(["funny lol", "lets football", "food is ready","food"])
# X1_indices = char_to_indices(X1,char_to_ix, max_len = max_len)
# print("X1 =", X1)
# print("X1_indices =", X1_indices)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2,stratify=y)


X_train_indices = char_to_indices(X_train, char_to_ix, max_len)
Y_train_oh = convert_to_one_hot(y_train, C = 2)
X_modified_train = np.reshape(X_train_indices, (len(X_train_indices), max_len, 1))

X_test_indices = char_to_indices(X_test, char_to_ix, max_len)
Y_test_oh = convert_to_one_hot(y_test, C = 2)
X_modified_test = np.reshape(X_test_indices, (len(X_test_indices), max_len, 1))


model = Sequential()
model.add(LSTM(300, input_shape=(X_modified_train.shape[1], X_modified_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))


print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_modified_train,Y_train_oh, epochs = 10, validation_data=(X_modified_test,Y_test_oh))

model.save('char_level.h5')


# def char_level_lstm(input_shape):
#
#     sentence_indices = Input(input_shape, dtype='int32')
#
#     # embeddings = embedding_layer(sentence_indices)
#
#     # X = LSTM(128, return_sequences=True)(embeddings)
#     X = LSTM(128, return_sequences=True)(sentence_indices)
#     X = Dropout(0.5)(X)
#     X = LSTM(128, return_sequences=False)(X)
#     X = Dropout(0.5)(X)
#     # no of output classes = 2
#     X = Dense(2)(X)
#     X = Activation('softmax')(X)
#
#     model = Model(inputs=sentence_indices, outputs=X)
#
#     return model
#
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state=2,stratify=y)
#
# model = char_level_lstm((max_len,))
# print(model.summary())
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# X_train_indices = char_to_indices(X_train, char_to_ix, max_len)
# Y_train_oh = convert_to_one_hot(y_train, C = 2)
#
# model.fit(X_train_indices,Y_train_oh, epochs = 10, batch_size = 32, shuffle=True)
#
# X_test_indices = char_to_indices(X_test, char_to_ix, max_len)
# Y_test_oh = convert_to_one_hot(y_test, C = 2)
# loss, acc = model.evaluate(X_test_indices, Y_test_oh)
# print()
# print("Test accuracy = ", acc)