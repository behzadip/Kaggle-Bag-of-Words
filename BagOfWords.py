# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.ensemble import RandomForestClassifier


os.chdir(os.path.dirname(os.path.abspath(__file__)))
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter = "\t", quoting=3)
test = pd.read_csv('testData.tsv', header=0, delimiter = "\t", quoting=3)

stopwords = set(nltk.corpus.stopwords.words("english"))
def parser(raw_text):
    text = BeautifulSoup(raw_text, "lxml").get_text()
    all_words = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    words = [w for w in all_words if not w in stopwords]
    return (" ".join(words))
reviews = [parser(review) for review in train["review"]]

vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,    \
    preprocessor = None, stop_words = None, max_features = 5000)

train_features = vectorizer.fit_transform(reviews).toarray()
vocab = vectorizer.get_feature_names()
freq = np.sum(train_features, axis=0)

for tag, count in zip(vocab, freq):
    print count, tag

forest = RandomForestClassifier(n_estimators = 100) 
forest.fit(train_features, train["sentiment"] )


reviews_test = [parser(review) for review in test["review"]]
test_features = vectorizer.transform(reviews_test).toarray()
result = forest.predict(test_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "submission-Bag_of_Words.csv", index=False, quoting=3 )

   