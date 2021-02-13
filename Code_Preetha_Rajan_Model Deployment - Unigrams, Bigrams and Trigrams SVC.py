#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# In[2]:


def normalizer(rev): #### Cleaning Reviews
    soup = BeautifulSoup(rev, 'lxml')   # removing HTML encoding such as ‘&amp’,’&quot’ or html tags such as '&gt'
    souped = soup.get_text()
    re1 = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", " ", souped) # removing @mentions, urls
    re2 = re.sub("[^A-Za-z]+"," ", re1) # removing numbers

    tokens = nltk.word_tokenize(re2)
    removed_letters = [word for word in tokens if len(word)>2] # removing words with length less than or equal to 2
    lower_case = [l.lower() for l in removed_letters]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t, pos='v') for t in filtered_result]
    return lemmas


# In[3]:


def main():
    #### Loading the saved model and its vocabulary
    model = joblib.load('svc-unigrams, bigrams and trigrams.sav')
    vocabulary_model = pd.read_csv('vocabulary - unigrams, bigrams and trigrams.csv', header=None)
    vocabulary_model_dict = {}
    for i, word in enumerate(vocabulary_model[0]):
         vocabulary_model_dict[word] = i
    tfidf = TfidfVectorizer(vocabulary = vocabulary_model_dict, min_df=30, ngram_range=(1,3)) # min_df=5 is clever way of feature engineering
    #### Reading the test_reviews.csv dataset
    df = pd.read_csv('test_reviews.csv', encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
    #### Normalizing reviews
    df['normalized_reviews'] = df.extract.apply(normalizer)
    df = df[df['normalized_reviews'].map(len) > 0] # removing rows with normalized reviews of length 0
    print("Number of reviews remained after cleaning: ", df.normalized_reviews.shape[0])
    print(df[['extract','normalized_reviews']].head())
    #### Saving cleaned reviews to csv file
    df.drop(['source', 'domain', 'extract','product'], axis=1, inplace=True)
    df.to_csv('cleaned_reviews.csv', encoding='utf-8', index=False)
    cleaned_reviews = pd.read_csv("cleaned_reviews.csv", encoding = "ISO-8859-1")
    cleaned_reviews_tfidf = tfidf.fit_transform(cleaned_reviews['normalized_reviews'])
    targets_pred = model.predict(cleaned_reviews_tfidf)
    #### Saving predicted rating of reviews to csv
    cleaned_reviews['predicted_rating'] = targets_pred.reshape(-1,1)
    cleaned_reviews.to_csv('Predicted Ratings – Unigrams, Bigrams and Trigrams SVC.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()


# In[ ]:




