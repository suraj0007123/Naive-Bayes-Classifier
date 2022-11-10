import pandas as pd
import numpy as np


tweet=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\naive bayes classifier\Datasets_Naive Bayes\Disaster_tweets_NB.csv",encoding="ISO-8859-1")

tweet.head()

tweet.shape

tweet.dtypes

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re

stop_words=[]

with open(r"E:\DESKTOPFILES\suraj\Datasets NLP\stop.txt","r") as sw:
    stop_words=sw.read()
    
stop_words=stop_words.split("\n")

def cleaning_text(i):
    i=re.sub("[^A-Za-z" "]+"," ",i).lower()
    i=re.sub("[0-9" "]+"," ",i)
    w=[]
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return(" ".join(w))

### Testing above function with sample text=> removes punctuations, numbers 

cleaning_text("Hope you are having a good week. Just checking in")
cleaning_text("hope i can understand your feelings 123121. 123 hi how .... are you?")
cleaning_text("Hi how are you, I am good")


tweet.text=tweet.text.apply(cleaning_text)


###### Removing empty rows 
tweet = tweet.loc[tweet.text !=" ",:]

from sklearn.model_selection import train_test_split
tweet_train, tweet_test = train_test_split(tweet, test_size = 0.2)

### Creating a matrix of token counts for the entire text document

def split_into_words(i):
    return[word for word in i.split(" ")]


# Defining the preparation of tweet texts into word count matrix format - Bag of Words
tweet_bow = CountVectorizer(analyzer=split_into_words).fit(tweet.text)

# Apply BOW for all messages
all_tweet_matrix=tweet_bow.transform(tweet.text)

# For training messages
train_tweet_matrix=tweet_bow.transform(tweet_train.text)

# For testing messages
test_tweet_matrix = tweet_bow.transform(tweet_test.text)

# Learning Term weighting and normalizing on entire tweet
tfidf_transformer=TfidfTransformer().fit(all_tweet_matrix)

# Preparing TFIDF for train tweet
train_tfidf=tfidf_transformer.transform(train_tweet_matrix)

train_tfidf.shape # (row, column)

# Preparing TFIDF for test tweet
test_tfidf = tfidf_transformer.transform(test_tweet_matrix)

test_tfidf.shape   #  (row, column) 

# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
print(accuracy_test_m)

#another method to find the accuracy usin "accuracy_score"
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, tweet_test.target) 

pd.crosstab(test_pred_m, tweet_test.target)

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
print(accuracy_score(test_pred_lap,tweet_test.target))

print(pd.crosstab(test_pred_lap, tweet_test.target))

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweet_train.target)
accuracy_train_lap


