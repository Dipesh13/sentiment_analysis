#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# filename = "sample.txt"
filename = "Frankenstein.txt"

text = (open(filename).read()).lower()

# mapping characters with integers
unique_chars = sorted(list(set(text)))

char_to_int = {}
int_to_char = {}

for i, c in enumerate (unique_chars):
    char_to_int.update({c: i})
    int_to_char.update({i: c})

X = []
Y = []

max_len = 100

for i in range(0, len(text) - max_len, 1):
    sequence = text[i:i + max_len]
    label =text[i + max_len]
    X.append([char_to_int[char] for char in sequence])
    Y.append(char_to_int[label])

# for a,b in zip(X,Y):
#     print(a,b)

# reshaping, normalizing and one hot encoding
X_modified = np.reshape(X, (len(X), max_len, 1))
X_modified = X_modified / float(len(unique_chars))
Y_modified = np_utils.to_categorical(Y)

# print(X_modified.shape)
# print(X_modified.shape[1],X_modified.shape[2])


# defining the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# fitting the model
model.fit(X_modified, Y_modified, epochs=3, batch_size=30)

# picking a random seed
start_index = np.random.randint(0, len(X)-1)
new_string = X[start_index]

# generating characters
pred_sent = []
for i in range(max_len):
    x = np.reshape(new_string, (1, len(new_string), 1))
    x = x / float(len(unique_chars))

    #predicting
    pred_index = np.argmax(model.predict(x, verbose=0))
    char_out = int_to_char[pred_index]
    seq_in = [int_to_char[value] for value in new_string]
    print(char_out)
    pred_sent.append(char_out)

    new_string.append(pred_index)
    new_string = new_string[1:len(new_string)]


model.save('char_gen.h5')