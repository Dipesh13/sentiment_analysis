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


df = pd.read_csv('train.csv')

# sentences = df['data'].tolist()
# X = df['data']
# y = df['label']
# y  = pd.get_dummies(y)

# convert to numpy array
data_list = df['data'].tolist()
label_list = df['label'].tolist()
X = np.asarray(data_list)
y = np.asarray(label_list)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2,stratify=y)

maxLen = len(max(X_train, key=len).split())
print(maxLen)

# index = 20
# print(X_train[index],y_train[index])


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def sentence_to_avg(sentence, word_to_vec_map):

    words = [i.lower() for i in sentence.split()]

    avg = np.zeros((50,))

    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)

    return avg


def sentences_to_indices(X, word_to_index, max_len):

    m = X.shape[0]

    X_indices = np.zeros((m, max_len))

    for i in range(m):
        sentence_words = [w.lower() for w in X[i].split()]
        j = 0
        for w in sentence_words:
            # X_indices[i, j] = word_to_index[w]
            if w in word_to_index.keys():
                X_indices[i, j] = word_to_index[w]
            else :
                X_indices[i, j] = word_to_index['unk']
            j += 1

    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):

    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def word_lstm(input_shape, word_to_vec_map, word_to_index):

    sentence_indices = Input(input_shape, dtype='int32')

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentence_indices)

    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    # no of output classes = 2
    X = Dense(2)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('/home/mohit/Desktop/git/glove.6B.50d.txt')


# print(word_to_index.keys())
# if 'unk' in word_to_index.keys():
#     print(word_to_index['unk'])
# print(X.shape)

# word = "cucumber"
# index = 289846
# print("the index of", word, "in the vocabulary is", word_to_index[word])
# print("the", str(index) + "th word in the vocabulary is", index_to_word[index])
#
# avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
# print("avg = ", avg)
#
# X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
# X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
# print("X1 =", X1)
# print("X1_indices =", X1_indices)

# embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
# print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])

model = word_lstm((maxLen,), word_to_vec_map, word_to_index)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(y_train, C = 2)

model.fit(X_train_indices,Y_train_oh, epochs = 10, batch_size = 32, shuffle=True)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(y_test, C = 2)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

model.save('word_level.h5')
