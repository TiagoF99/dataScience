import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


my_data = pd.read_csv("drug200.csv", delimiter=",")

# shape is (200, 6)
print(my_data.head(), my_data.shape)

# convert Sex and Cholesterol into number values
my_data['Sex'] = pd.get_dummies(my_data['Sex'])
my_data['Cholesterol'] = pd.get_dummies(my_data['Cholesterol'])

# Let X be the feature matrix without the last column which is target
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# convert BP into numerical values (0,1,2)
le_BP = preprocessing.LabelEncoder()
print(le_BP)
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

print(X[:5])

# let y be target variable column
y = my_data["Drug"]
print(y.head())

# split into 30% rows for testing and 7-% of rows for training
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# specify criterion="entropy" so we can see information gain of each node
# max_depth=4 because 5 feature variables?
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
# it shows the default parameters
print(drugTree)

drugTree.fit(X_trainset,y_trainset)

# ========================================================
# PREDICTION

predTree = drugTree.predict(X_testset)

print(predTree[0:5])
print(y_testset[0:5])

# accuracy is 98.333333%
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# ============================================================
# DATA VISUALIZATION

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out = tree.export_graphviz(drugTree, feature_names=featureNames,
                         out_file=dot_data, class_names=np.unique(y_trainset),
                         filled=True,  special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
