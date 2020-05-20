from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from Data_Handling import *
import numpy as np 


def train_model():
    Data = import_data("Trian.csv")
    vectorizer = TfidfVectorizer(min_df= 2)

    all_docs_list = [row['Tweet_words'] for index, row in Data.iterrows()]

    all_docs = []
    for wl in all_docs_list:
        doc_string = ""
        for w in wl:
            doc_string = doc_string + w + " "
        all_docs.append(doc_string)

    tfidf = vectorizer.fit_transform(all_docs)
    tfidf = tfidf.toarray()
    print(tfidf.shape)

    y = [int(row['label']) for index, row in Data.iterrows()]

    y = np.array(y)

    x = tfidf

    lgr = LogisticRegression(class_weight='balanced').fit(x, y)

    return lgr
