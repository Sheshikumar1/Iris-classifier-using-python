# installing libraries

import sklearn
import numpy
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

### by using dataset of iris.csv

df=pd.read_csv("iris.csv")
df.head()

df.describe()

cdf=df[['sepal-length','sepal-width','petal-length','petal-width']]
cdf.head(10)

# Analysis 

df.info()

df['species'].value_counts()

df.isnull().sum()

# Visualization of data

df['sepal-length'].hist()

df['sepal-width'].hist()



df['petal-length'].hist()

df['petal-width'].hist()

sns.pairplot(df,hue='species')

colors = ['red', 'purple', 'green']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal-length'], x['sepal-width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()

for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['petal-length'], x['petal-width'], c = colors[i], label=species[i])
plt.xlabel("petal Length")
plt.ylabel("petal Width")
plt.legend()

for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal-length'], x['petal-length'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("petal length")
plt.legend()

for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal-width'], x['petal-width'], c = colors[i], label=species[i])
plt.xlabel("Sepal width")
plt.ylabel("petal Width")
plt.legend()

# Scatter Matrix

from sklearn.preprocessing import LabelEncoder
df.species=LabelEncoder().fit_transform(df.species)

from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(12,12))
plt.show()

# Correlation

sns.heatmap(df.corr(),annot=True)



# Modeling Training And testing:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X=df.drop(columns=['species'])
Y=df['species']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)


#logistric Regression
model=LogisticRegression()
model.fit(X_train,Y_train)
print("Logistic Regreesion Accuracy:",model.score(X_test,Y_test)*100)

## model training

model.fit(X_train.values,Y_train.values)

#print metric to get performance
print("Accuracy:",model.score(X_test,Y_test)*100)

#K-nearest neighbors
model=KNeighborsClassifier()
model.fit(X_train.values,Y_train.values)
print("K-nearest neighbors Accuracy:",model.score(X_test,Y_test)*100)


model.fit(X_train.values,Y_train.values)

#print metric to get performanace
print("Accuracy:",model.score(X_test,Y_test)*100)

#Decision tree
model=DecisionTreeClassifier()
model.fit(X_train.values,Y_train.values)
print("Decision Tree Accuracy:",model.score(X_test,Y_test)*100)


model.fit(X_train.values,Y_train.values)

#print metric to get performanace
print("Accuracy:",model.score(X_test,Y_test)*100)

# Predicting the model

import pickle

filename='saved_model.sav'
try:
    with open(filename,'wb') as file:
        pickle.dump(model,file)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")

load_model=pickle.load(open(filename,'rb'))


load_model.predict([[6.0,2.2,4.0,1.0]])

X_test.head()

load_model.predict([[4,2.5,3.0,4.0]])

