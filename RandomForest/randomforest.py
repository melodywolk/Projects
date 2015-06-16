import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import os
from fnmatch import fnmatch
import math


def def_to_words( raw_text ):
    # Function to convert a raw review to a string of words
    # The input is a single string, and 
    # the output is a single string (a preprocessed text)
    #
    # 1. Remove HTML
    text = BeautifulSoup(raw_text).get_text()
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


dir = "/home/melody/Desktop/NaiveBayes/samples/"
data = {
    "text": [],
    "category": []
}

for file in os.listdir(dir+"cryptids/"):
    text = open(dir+"cryptids/"+file).read()
    word = def_to_words(text)
    data["text"].append(word)
    data["category"].append(1)

for file in os.listdir(dir+"true/"):
    text = open(dir+"true/"+file).read()
    word = def_to_words(text)
    data["text"].append(word)             
    data["category"].append(0)  


type(data["text"])
from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 500) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(data["text"])

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

print " "
print train_data_features.shape
# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab[0:10]

print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the cryptid/true labels as the response variable

forest = forest.fit( train_data_features, data["category"] )

test = {
    "text": []
}

for file in os.listdir("./test/"):
    print file
    text = open("./test/"+file).read()
    word = def_to_words(text)
    test["text"].append(word)


test_data_features = vectorizer.transform(test["text"])
test_data_features = test_data_features.toarray()

# Use the random forest to make label predictions
result = forest.predict(test_data_features)

print result
