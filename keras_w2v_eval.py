#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import string
from keras.models import load_model
from get_embedding import sent_embedding

df = pd.read_csv('test_oJQbWVk.csv')
# Pre-Processing : remove punctations.
# digits lower case in sent_embediing.
# stop words left. use nltk for that
# decode left
df['tweet']= df['tweet'].apply(lambda x: x.translate(None, string.punctuation))
df['tweet']= df['tweet'].apply(lambda x: x.translate(None, string.digits))
# df['data']= df['data'].apply(lambda x: x.decode('utf-8'))
sentences = df['tweet'].tolist()


X = []
for sentence in sentences:
    sent_emb = sent_embedding(sentence)
    X.append(sent_emb)

clf = load_model('keras_w2v_300_final.h5')

print(clf.predict_classes(np.array(X)))

# output_labels = clf.predict_classes(np.array(X))
#
# df = pd.DataFrame(
#     {'id': df['id'],
#      'label': output_labels,
#     })
#
# print(df)
# df.to_csv('solution.csv',index=False)