#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from get_embedding import sent_embedding

df = pd.read_csv('train_2kmZucJ.csv')
df['tweet']= df['tweet'].apply(lambda x: x.translate(None, string.punctuation))
df['tweet']= df['tweet'].apply(lambda x: x.translate(None, string.digits))
# df['data']= df['data'].apply(lambda x: x.decode('utf-8'))
sentences = df['tweet'].tolist()
y = df['label'].tolist()


X = []
for sentence in sentences:
    sent_emb = sent_embedding(sentence)
    X.append(sent_emb)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2,stratify=y)

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()
]

names = ["Logistic Regression",
         "Nearest Neighbors",
         "Linear SVM",
         "RBF SVM",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost"
         "Naive Bayes"
         ]

model_names = dict(zip(names,classifiers))

for key,val in model_names.items():
    pl = Pipeline([
        ('clf',val)
    ])

    pl.fit(X_train,y_train)

    preds = pl.predict(X_train)
    print(key + " train accuracy: ", accuracy_score(y_train, preds))
    preds_test = pl.predict(X_test)
    print(key + " test accuracy: ", accuracy_score(y_test, preds_test))

    # with open(key+'.pickle', 'wb') as fo:
    #     pickle.dump(pl,fo)
