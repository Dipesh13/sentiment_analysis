import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath,get_tmpfile
from nltk import word_tokenize

glove_file = datapath('/home/dipesh/Downloads/glove.6B.50d.txt')
tmp_file = get_tmpfile("glove_word2vec.txt")
glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)

def sent_embedding(sentence,model=model):
    #  add check for 1) empty sentence 2) sentence containing all words which are out of vocab.
    tokens = [w for w in word_tokenize(sentence.lower()) if w.isalpha()]
    sent_emb = np.mean([model[t] if t in model else model['unk'] for t in tokens ],axis=0)
    return sent_emb
# print(sent_emb)