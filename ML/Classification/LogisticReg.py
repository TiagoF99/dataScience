import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss

churn_df = pd.read_csv("ChurnData.csv")
print(churn_df.head())

# reduce columns on churn_df
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
# change column to type int
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])
# normalize the data
X = preprocessing.StandardScaler().fit(X).transform(X)

# split model into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to
# solve the overfitting problem in machine learning models. C parameter indicates inverse of regularization strength
# which must be a positive float.
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat_prob = LR.predict_proba(X_test)
yhat = LR.predict(X_test)
print(yhat)

# predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is the
# probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

# simularity of sample set -> intersection of union of A and B
print(jaccard_score(y_test, yhat))

# probability of customer churn is yes
print(log_loss(y_test, yhat_prob))