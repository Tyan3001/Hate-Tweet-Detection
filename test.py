from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from Data_Handling import *
from LogisticRegression import *
import numpy as np 

test_data = import_data('test.csv')

vectorizer = TfidfVectorizer(min_df= 2)

all_docs_list = [row['Tweet_words'] for index, row in test_data.iterrows()]

all_docs = []
for wl in all_docs_list:
    doc_string = ""
    for w in wl:
        doc_string = doc_string + w + " "
    all_docs.append(doc_string)

tfidf = vectorizer.fit_transform(all_docs)
tfidf = tfidf.toarray()

x_test = tfidf
y_test = [int(row['label']) for index, row in test_data.iterrows()]

lgr = train_model()

y_pred = lgr.predict(x_test)

print(confusion_matrix(y_test, y_pred))