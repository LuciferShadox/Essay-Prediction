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
import time

def get_data():
    dataset=pd.read_excel('training_set_rel3.xls',parse_cols=6)
    return dataset

#bag of Words
def BOW(essay):
    vectorizer = CountVectorizer(max_features = 5000, ngram_range=(1, 3), stop_words='english')
    countvectors=(vectorizer.fit_transform(essay)).toarray()
    # to know the mapped words
    # featurenames=vectorizer.get_feature_names()
    return countvectors

#Parts of Speech (POS) count
def POS(essay):
    cleaned_essay=re.sub(r'\W', ' ', essay)
    words=nltk.word_tokenize(cleaned_essay)
    word_count=len(words)
    sentences=nltk.sent_tokenize(essay)
    sentences_count=len(sentences)
    avg_len_sent=0
    for sent in sentences:
        avg_len_sent+=len(sent)
    avg=avg_len_sent/sentences_count
    return word_count,sentences_count,avg_len_sent


start_time = time.time()
dataframe=get_data()
essay_set=dataframe[['essay']].copy()
score=dataframe['domain1_score']
#for testing purpose I decreased set
essay_set=essay_set[:1000]
score=score[:1000]
countvectors=BOW(essay_set['essay'])
essay_set['word_count'],essay_set['sent_count'],essay_set['avg_sent_count']=zip(*essay_set['essay'].apply(POS))
x=np.concatenate((essay_set.iloc[:,1:].values,countvectors),axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,score,random_state=0, test_size=0.20)


model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
#check mean square error
mse=mean_squared_error(y_test,y_pred)
print ("Mean Squared Error :",mse)
#Execution Time
print("--- %s seconds ---" % (time.time() - start_time))

#to save the model for future use
joblib.dump(model,'mod.pkl')
