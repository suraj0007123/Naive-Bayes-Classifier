import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

train_data=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\naive bayes classifier\Datasets_Naive Bayes\SalaryData_Train.csv")

test_data=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\naive bayes classifier\Datasets_Naive Bayes\SalaryData_Test.csv")

train_data.head()

test_data.head()

##### Train data
#### Under all the columns select only object columns which are included the non-numeric
non_numeric=train_data.select_dtypes(include=['object'])

#### Under all the columns select only numeric columns which are excluded the numeric
numeric = train_data.select_dtypes(exclude=["object"])

######### Label Encoder ############
from sklearn.preprocessing import LabelEncoder

##### creating instance of labelencoder #####
labelencoder=LabelEncoder()

non_numeric.iloc[:,0]=labelencoder.fit_transform(non_numeric.iloc[:,0])

non_numeric.iloc[:,1]=labelencoder.fit_transform(non_numeric.iloc[:,1])

non_numeric.iloc[:,2]=labelencoder.fit_transform(non_numeric.iloc[:,2])

non_numeric.iloc[:,3]=labelencoder.fit_transform(non_numeric.iloc[:,3])

non_numeric.iloc[:,4]=labelencoder.fit_transform(non_numeric.iloc[:,4])

non_numeric.iloc[:,5]=labelencoder.fit_transform(non_numeric.iloc[:,5])

non_numeric.iloc[:,6]=labelencoder.fit_transform(non_numeric.iloc[:,6])

non_numeric.iloc[:,7]=labelencoder.fit_transform(non_numeric.iloc[:,7])

non_numeric.iloc[:,8]=labelencoder.fit_transform(non_numeric.iloc[:,8])

#### Concatenating the numeric and non_numeric columns together
train=pd.concat([numeric,non_numeric],axis=1)

train.head()

train.shape

##### Test data
#### Under all the columns select only object columns which are included the non-numeric
non_numeric=test_data.select_dtypes(include=['object'])

#### Under all the columns select only numeric columns which are excluded the numeric
numeric = test_data.select_dtypes(exclude=["object"])

######### Label Encoder ############
from sklearn.preprocessing import LabelEncoder

##### creating instance of labelencoder #####
labelencoder=LabelEncoder()

non_numeric.iloc[:,0]=labelencoder.fit_transform(non_numeric.iloc[:,0])

non_numeric.iloc[:,1]=labelencoder.fit_transform(non_numeric.iloc[:,1])

non_numeric.iloc[:,2]=labelencoder.fit_transform(non_numeric.iloc[:,2])

non_numeric.iloc[:,3]=labelencoder.fit_transform(non_numeric.iloc[:,3])

non_numeric.iloc[:,4]=labelencoder.fit_transform(non_numeric.iloc[:,4])

non_numeric.iloc[:,5]=labelencoder.fit_transform(non_numeric.iloc[:,5])

non_numeric.iloc[:,6]=labelencoder.fit_transform(non_numeric.iloc[:,6])

non_numeric.iloc[:,7]=labelencoder.fit_transform(non_numeric.iloc[:,7])

non_numeric.iloc[:,8]=labelencoder.fit_transform(non_numeric.iloc[:,8])

#### Concatenating the numeric and non_numeric columns together
test=pd.concat([numeric,non_numeric],axis=1)

test.head()

test.shape

train_x=train.iloc[:,:13]
train_y=train.iloc[:,13]
test_x=test.iloc[:,:13]
test_y=test.iloc[:,13]

train_x.columns
test_x.columns

from sklearn.naive_bayes import MultinomialNB as MB

## Creating the multinomial naive bayes function
classifier_mb = MB()

classifier_mb.fit(train_x,train_y)

pred=classifier_mb.predict(test_x)

from sklearn.metrics import accuracy_score

print("Accuracy Score:",accuracy_score(test_y,pred))

pd.crosstab(test_y,pred,rownames=["Actual"],colnames=["Predicted"])

pred_x=classifier_mb.predict(train_x)

accuracy_score(pred_x,train_y)
