import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv('teleCust1000t.csv')
df.head()

# how many of each subgroup there are
df['custcat'].value_counts()

# in format [[x,x,x,x,x,x,x,x,x,x,x], [x,x,x,x,x,x,x,x,x,x,x], ....]
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
# normalize the data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

y = df['custcat'].values

# X_train is 80% of rows of X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

k = 4
# Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# we can use model to predict values
yhat = neigh.predict(X_test)

# notice that anything involving y are the actual 'custcat' values
# Notice both are being tested against the train fitted model
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# ==========================================================================================================
# HERE IS HOW TO TEST THE VALUE OF K
Ks = 10
mean_acc = np.zeros((Ks - 1))

# BEST VALUE OF K is mean_acc.max() so k= index of mean_acc.max()
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

