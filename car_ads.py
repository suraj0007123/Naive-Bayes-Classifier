import pandas as pd
import numpy as np

car_data=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\naive bayes classifier\Datasets_Naive Bayes\NB_Car_Ad.csv")

car_data.head()

car_data.shape

######### LabelEncoder ######
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

car_data["Gender"]=labelencoder.fit_transform(car_data["Gender"])

car_data

car_data.shape

X=car_data.iloc[:,:4]
Y=car_data.iloc[:,4]

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.20)

from sklearn.naive_bayes import MultinomialNB as MB
#### creating mutlinomial Naive Bayes function 

classifier_mb=MB()

classifier_mb.fit(train_x,train_y)

pred=classifier_mb.predict(test_x)

from sklearn.metrics import accuracy_score

print("Accuracy Score:",accuracy_score(test_y,pred))

pd.crosstab(test_y,pred,rownames=["Actual"], colnames=["predicted"])


pred_x=classifier_mb.predict(train_x)

accuracy_score(pred_x,train_y)

