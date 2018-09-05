#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
# https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/


# CHAR LEVEL
# create a voacbulary of all chars in input dataset
data = open('data.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

# create char_to_index and index_to_char map
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
print(ix_to_char)


# WORD LEVEL
# https://github.com/HeroKillerEver/coursera-deep-learning/blob/master/Sequence%20Models/Emojify/emo_utils.py
glove_file = 'glove.6B.50d.txt'

# word_to_vec_map = read_glove_vecs()
with open(glove_file, 'r') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        # word_to_vec_map[curr_word] = model[word]

# word to index and vice versa
i = 1
words_to_index = {}
index_to_words = {}
for w in sorted(words):
    words_to_index[w] = i
    index_to_words[i] = w
    i = i + 1



# # Preparing dataset
# # Data is prepared in a format such that if we want the LSTM to predict the ‘O’ in ‘HELLO’
# # we would feed in [‘H’, ‘E‘ , ‘L ‘ , ‘L‘ ] as the input and [‘O’] as the expected output.
# # Similarly, here we fix the length of the sequence that we want (set to 50 in the example)
# # and then save the encodings of the first 49 characters in X and the expected output
# # i.e. the 50th character in Y.
#
# # preparing input and output dataset
# X = []
# Y = []
#
# for i in range(0, len(text) - 50, 1):
#     sequence = text[i:i + 50]
#     label =text[i + 50]
#     X.append([char_to_int[char] for char in sequence])
#     Y.append(char_to_int[label])
#
#
# # reshaping, normalizing and one hot encoding
# X_modified = np.reshape(X, (len(X), 50, 1))
# X_modified = X_modified / float(len(unique_chars))
# Y_modified = np_utils.to_categorical(Y)