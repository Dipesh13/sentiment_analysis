#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
from get_embedding import sent_embedding

with open('K Nearest Neighbours.pickle', 'rb') as fi:
    model = pickle.load(fi)

def prediction(data):
    for sentence in data:
        sent_emb = sent_embedding(sentence)
        label = model.predict(sent_emb.reshape(1, -1))
        return label[0]