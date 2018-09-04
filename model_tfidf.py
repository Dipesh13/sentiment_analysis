#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score
import string
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('train_2kmZucJ.csv')
# digits lower case in sent_embediing.
# decode left
df['tweet']= df['tweet'].apply(lambda x: x.translate(None, string.punctuation))
X= df['tweet'].apply(lambda x: x.translate(None, string.digits))
y= df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2,stratify=y)

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1)
    # AdaBoostClassifier(),
    # GaussianNB()
]

names = ["Logistic Regression",
         "Nearest Neighbors",
         "Linear SVM",
         "RBF SVM",
         "Decision Tree",
         "Random Forest",
         "Neural Net"
         # "AdaBoost"
         # "Naive Bayes"
         ]

model_names = dict(zip(names,classifiers))

for key,val in model_names.items():
    pl = Pipeline([
        ('vectorizer',TfidfVectorizer(stop_words='english',ngram_range=(1,3),min_df=3,max_df=100,max_features=None)),
        ('clf',val)
    ])

    pl.fit(X_train,y_train)

    preds = pl.predict(X_train)
    print(key + " train accuracy: ", accuracy_score(y_train, preds))
    preds_test = pl.predict(X_test)
    print(key + " test accuracy: ", accuracy_score(y_test, preds_test))

    # with open(key+'_tfidf.pickle', 'wb') as fo:
    #     pickle.dump(pl,fo)

