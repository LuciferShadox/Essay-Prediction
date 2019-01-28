# importing necessary libraries 
import numpy as np  # numpy for matrix operations
import pandas as pd  # for file handling
import matplotlib.pyplot as plt
import re,collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


def get_data(filename):
    dataset=pd.read_excel(filename,parse_cols=6)
    return dataset

def BOW(essay):
    vectorizer = CountVectorizer(max_features = 5000, ngram_range=(1, 3), stop_words='english')
    countvectors=(vectorizer.fit_transform(essay)).toarray()
    # to know the mapped words
    # featurenames=vectorizer.get_feature_names()
    return countvectors

dataframe=get_data('training_set_rel3.xls')
essay_set=dataframe['essay']
score=dataframe['domain1_score']
#for testing purpose I decreased set
essay_set=essay_set[:800]
score=score[:800]
countvectors=BOW(essay_set)
x_train,x_test,y_train,y_test=train_test_split(countvectors,score,random_state=0, test_size=0.20)
model=LinearRegression()
model.fit(x_train,y_train)
#to save the model for future use
joblib.dump(model,'mod.pkl')
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print ("Mean Squared Error :",mse)
