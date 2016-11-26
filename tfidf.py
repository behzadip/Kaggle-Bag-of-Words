# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
import nltk.data
import re
from sklearn.feature_extraction.text import CountVectorizer 

os.chdir(os.path.dirname(os.path.abspath(__file__)))
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter = "\t", quoting=3)
test = pd.read_csv('testData.tsv', header=0, delimiter = "\t", quoting=3)
un_train = pd.read_csv('unlabeledTrainData.tsv', header=0, delimiter = "\t", quoting=3)

stopwords = set(nltk.corpus.stopwords.words("english"))
def parser(raw_text, remove_stopwords=False):
    text = BeautifulSoup(raw_text, "lxml").get_text()
    all_words = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower().split()
    #words = [w for w in all_words if not w in stopwords]
    if remove_stopwords:
        all_words = [w for w in all_words if not w in stopwords]    
    return (all_words)
#reviews = [parser(review) for review in train["review"]]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

train_reviews = []
for review in train["review"]:
    train_reviews.append(" ".join(parser(review, remove_stopwords=False)))

test_reviews = []
for review in test["review"]:
    test_reviews.append(" ".join(parser(review, remove_stopwords=False)))

un_train_reviews = []
for review in un_train["review"]:
    un_train_reviews.append(" ".join(parser(review, remove_stopwords=False)))


print "Vectorizing..."

vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, max_features = 200000, ngram_range = (1, 2), sublinear_tf = True)

vectorizer = vectorizer.fit(train_reviews + un_train_reviews)
train_data_features = vectorizer.transform(train_reviews)
test_data_features = vectorizer.transform(test_reviews)
'''
print "Reducing dimension..."

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
fselect = SelectKBest(chi2 , k=70000)
train_data_features = fselect.fit_transform(train_data_features, train["sentiment"])
test_data_features = fselect.transform(test_data_features)

print "Training..."

model1 = MultinomialNB(alpha=0.0005)
model1.fit( train_data_features, train["sentiment"] )

model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model2.fit( train_data_features, train["sentiment"] )

p1 = model1.predict_proba( test_data_features )[:,1]
p2 = model2.predict_proba( test_data_features )[:,1]

print "Writing results..."

output = pd.DataFrame( data = { "id": test["id"], "sentiment": .2*p1 + 1.*p2 } )
output.to_csv("submission-tfidf2.csv", index = False, quoting = 3 )
'''

print 'vectorizing... ', 
tfv = TfidfVectorizer(min_df=2,  max_features=None, 
        strip_accents='unicode', analyzer='word',
        ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
X_all = train_reviews + test_reviews
lentrain = len(train_reviews)

print "fitting pipeline... ",
tfv.fit(X_all)
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)
print "20 Fold CV Score: ", np.mean(cross_val_score(model, X, train['sentiment'], cv=20, scoring='roc_auc'))

print "Retrain on all training data, predicting test labels...\n"
model.fit(X,train['sentiment'])
result = model.predict_proba(X_test)[:,1]
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv("submission-tfidf.csv", index=False, quoting=3)
#output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)

