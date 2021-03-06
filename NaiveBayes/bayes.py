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
    # Function to convert a text to a string of words
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
    # 6. Return the result.
    return meaningful_words  

def count_words(words):
    wc = {}
    for word in words:
         wc[word] = wc.get(word, 0.0) + 1.0
        
    return wc


s = "You know nothing Jon Snow. Winter is coming."
print count_words(def_to_words(s))


vocab = {}
word_counts = {
    "cryptid": {},
    "real": {}
}
priors = {
    "cryptid": 0.,
    "real": 0.
}
docs = []

dir = "/home/melody/Desktop/NaiveBayes/samples/"
category = "cryptid"
for file in os.listdir(dir+"cryptids/"):
    priors[category] +=1
    text = open(dir+"cryptids/"+file).read()
    word = def_to_words(text)
    counts = count_words(word)
    for w, count in list(counts.items()):
        if w not in vocab:
           vocab[w] = 1
        if w not in word_counts[category]:
           word_counts[category][w] = 1

    vocab[w] += count
    word_counts[category][w] += count

category = "real"
for file in os.listdir(dir+"true/"):
    priors[category] +=1
    text = open(dir+"true/"+file).read()
    word = def_to_words(text)
    counts = count_words(word)
    for w, count in list(counts.items()):
        if w not in vocab:
           vocab[w] = 1
        if w not in word_counts[category]:
           word_counts[category][w] = 1

    vocab[w] += count
    word_counts[category][w] += count

#new_doc = open("./test/Yeti.txt").read()
new_doc = open("./test/Dog.txt").read()
words = def_to_words(new_doc)
counts = count_words(words)

prior_real = (priors["real"] / sum(priors.values()))
prior_cryptid = (priors["cryptid"] / sum(priors.values()))

print priors["real"], priors["cryptid"]

prob_real = 0
prob_cryptid = 0
for w, count in list(counts.items()):
    # skip words that we haven't seen before, or words less than 3 letters long
    if w not in vocab:
        continue

    p_word = vocab[w] / sum(vocab.values())
    p_w_given_real = word_counts["real"].get(w, 0.0) / sum(word_counts["real"].values())
    p_w_given_cryptid = word_counts["cryptid"].get(w, 0.0) / sum(word_counts["cryptid"].values())
    
    if(p_w_given_real > 0):
       prob_real += np.log(count * p_w_given_real/p_word)
    if(p_w_given_cryptid > 0):
       prob_cryptid += np.log(count * p_w_given_cryptid/p_word)

print("Score(real)  :", np.exp(np.log(prob_real) + np.log(prior_real)) )
print("Score(cryptid):", np.exp(np.log(prob_cryptid) + np.log(prior_cryptid)) )    


