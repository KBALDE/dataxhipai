#!/usr/bin/env python
# coding: utf-8



import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# ## 1 - Read reviews data using Yelp API
# 
# In this part, I will request for a given restaurant some reviews and save them in a csv file.
# 
# Since there is no data for Avis Restau, we are going to use Yelp data as a proxy for our our use case.
# 
# in this part, we are going to retrieve some data from Yelp through the Yelp API and store them in a CSV file.

# In[90]:


#read 
import pandas as pd
import numpy as np

from PIL import Image, ImageFilter, ImageOps
from io import BytesIO
import nltk
import json


import requests



import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk import PorterStemmer , WordNetLemmatizer, word_tokenize



def dfColumnToList(df, col):
    """args: take a df and its column 
       return a list of rows from df    
    """
    
    com=[]
    for t in df[col].values:
        com.append(t)
    return com
    
def wordTokenizeList(text_list):
    """
    arguments: 
      - text_list: a list containing text
    returns:
      - a list of tokens
    """
    A=[]
    B=[]
    #tokenize 
    for x in text_list:
        A.append(word_tokenize(x))
    # collapse the list
    for i in range(len(A)):
        B+=A[i]
    return B

stopwords=nltk.corpus.stopwords.words("english")

def wordAnalyze(text_token):
    # stemming
    engstem=EnglishStemmer()
    wordan_stem=[engstem.stem(x).lower() for x in text_token]
    
    # lemmatizing
    wolem=WordNetLemmatizer()
    wordan_lem=[wolem.lemmatize(x).lower() for x in text_token]
   
    #stopwords from the input text
    wordan_lem_filter=[e.lower() for e in wordan_lem if len(e)>=3]
    # dropout english reco stopwords
    words=[w for w in wordan_lem_filter  if w.lower() not in stopwords] 
    
    #word frequency distro
    fdist=nltk.FreqDist(words)
    
    return wordan_stem, wordan_lem, words, fdist


# the cutting is done with linux sed and cut command

filename="data/yelp_dataset/review5000.json"
df = pd.read_json(filename, lines=True)


# filter to select 1 and 2 stars
df=df.loc[df.stars <=2]




# take the text column
text_list=dfColumnToList(df, 'text')

# Tokenization
text_token=wordTokenizeList(text_list)



# this function take a tokenized text and return Stemmed, Lemmatized, stopword filtered and most common words
wordan_stem, wordan_lem, words, fdist=wordAnalyze(text_token)

# LDA

from LDA_utils import *

stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say',
                   'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try',
                   'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 
                   'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 
                   'even', 'also', 'may', 'take', 'come'])


data = df.text.values.tolist() # smart way to do it
data_words = list(sent_to_words(data))
#print(data_words[:1])



nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # en_core_web_sm instead of 'en'
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc))
                  if word not in stop_words] for doc in texts_out]    
    return texts_out

#data ready
data_ready=process_words(data_words, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



# Create Dictionary
id2word = corpora.Dictionary(data_ready)
#id2word = corpora.Dictionary(data_words)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)




#import pyLDAvis.gensim
#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
#vis

