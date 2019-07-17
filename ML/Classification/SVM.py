import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

"""
STEPS

1. clean the data
2. split into feature and target values
3. split data into train and test
4. fit the model with the training data
5. predict the output with X_test
6. test accuracy of prediction against y_test
"""

pd.set_option('display.max_columns', None)
cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head())

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

# drop all BareNuc things that arnt nums and then change column type to int
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)


# to access X or y do X[0:_] or y[0:_]
# set the x/feature variables
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
# set y to the target variable
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

"""
NOTICE THERE ARE DIFFERENT TYPES OF SVM ALGORITHMS:
1.Linear
2.Polynomial
3.Radial basis function (RBF)
4.Sigmoid

-> WE USUALLY TRY ALL FOUR AND SEE WHICH GIVES THE BEST RESULT
"""
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
# now the model is fit so we predict
yhat = clf.predict(X_test)

# Accuracy
# f1 is a measure of a tests accuracy
# jaccard index is simularity of sample set -> intersection of union of A and B
print(f1_score(y_test, yhat, average='weighted'))
print(jaccard_score(y_test, yhat))
# =====================================
# Check difference with linear model instead
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train)
yhat2 = clf2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2))

"""
FINAL CONCLUSION

notice that both model accuracys are approx. equal
-> model accuracy is very good (both sccuracy values close to 1)
"""