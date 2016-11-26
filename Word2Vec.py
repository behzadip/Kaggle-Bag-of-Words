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

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer):
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.decode("utf8").strip())
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence):
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(parser(raw_sentence, remove_stopwords=False))
    return sentences
    
sentences = [] 

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in un_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300                     
min_word_count = 35                      
num_workers = 4     
context = 10                                                                                         
downsampling = 1e-3  

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)

model.most_similar("man")
model.most_similar("queen")
model.most_similar("awful")

# Index2word is a list that contains the names of the words in 
# the model's vocabulary. Convert it to a set, for speed 
index2word_set = set(model.index2word)
# Vector Avergeing
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs



# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

train_reviews = []
for review in train["review"]:
    train_reviews.append(parser(review, remove_stopwords=False))

test_reviews = []
for review in test["review"]:
    test_reviews.append(parser(review, remove_stopwords=False))

trainDataVecs = getAvgFeatureVecs(train_reviews, model, num_features)
testDataVecs = getAvgFeatureVecs(test_reviews, model, num_features)

# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)

print "Fitting a random forest to labeled training data..."
forest = forest.fit(trainDataVecs, train["sentiment"] )

# Test & extract results 
result = forest.predict(testDataVecs)
# Write the test results 
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("submission-Word2Vec.csv", index=False, quoting=3)