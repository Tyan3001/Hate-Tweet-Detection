import pandas as pd 
import nltk as tk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


def import_data():

    neutral_words = ['user', "'s", '...', "n't", "'m", '``', "''", 'amp']
    def remove_punc(words) :
        return [w for w in words if w not in string.punctuation]

    Data = pd.read_csv("Train.csv")

    Data = Data.drop(['id'], axis = 1)

    tweets = list(Data.loc[ : , 'tweet'])

    def readAndCleanTweet(tweet):
        word_list = tk.word_tokenize(tweet)
        word_list = remove_punc(word_list)
        word_list = [w for w in word_list if w not in neutral_words]
        word_list = [w.lower() for w in word_list]
        stop_words = set(stopwords.words('english'))
        words = [w for w in word_list if w not in stop_words]
        pt = PorterStemmer()
        words = [pt.stem(w) for w in words]
        return words

    tweet_words = []
    for t in tweets:
        tweet_words.append(readAndCleanTweet(t))

    labels = list(Data.loc[ : , 'label'])

    df = pd.DataFrame({'label': labels, 'Tweet_words' : tweet_words})
    return df