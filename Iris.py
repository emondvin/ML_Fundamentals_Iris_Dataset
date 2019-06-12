import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from subprocess import check_output
#print(check_output("ls").decode("utf8"))

iris = pd.read_csv("Iris.csv")
    #load the dataset
iris.head(2)
    #show two first rows...
#iris.info()
    #check for any inconsistency in the data, look for null values
iris.drop('Id', axis=1, inplace=True)
    #drop the unnecessary ID column
    # axis=1 specifies it should be column wise
    #inplace=T means the changes should be reflected into the dataframe

fig = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange',label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
#fig.set_size_inches(10,6)
plt.show()
    #This shows relationship between sepal length and Width


fig1 = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='orange',label='Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue',label='Versicolor',ax=fig1)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green',label='Virginica',ax=fig1)
fig1.set_xlabel("Petal Length")
fig1.set_ylabel("Petal Width")
fig1.set_title("Petal Length vs Width")
fig1 = plt.gcf()
plt.show()
    #Scatter plot relationship between petal length and Width

iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
plt.show()

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species', y='PetalLengthCm', data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species', y='PetalWidthCm', data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species', y='SepalLengthCm', data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species', y='SepalWidthCm', data=iris)
    #violin plots to show density of length and width in the Species
    #Thinner = less density, fatter = higher density

# import packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression
    #for logistic regression algorithm
from sklearn.model_selection import train_test_split
    #use to split the data set for training and testing
from sklearn.neighbors import KNeighborsClassifier
    #for K nearest neighbours
from sklearn import svm
    #for Support Vector Machine Algorithm
from sklearn import metrics
    #for chekcing the model accuracy
from sklearn.tree import DecisionTreeClassifier
    #for using Decision tree algorithm

iris.shape #get shape of data set

plt.figure(figsize=(7,4))
sns.heatmap(iris.corr(), annot=True, cmap='cubehelix_r')
    #draws heatmap with input as the correlation matrix calculated by iris.corr
plt.show()
    # observe: Sepal width and length are not correlated
    # Petal width and length are highly correlated
train, test = train_test_split(iris, test_size = 0.3)
    #Our main data is split into train and test
    # test_size = 0.3 splits the data into 70% train and 30% test
# print(train.shape)
# print(test.shape)

train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']]
    #taking the training data features
train_y = train.Species
    # output of our training data
test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']]
    #take test data features
test_y = test.Species
    #output of test database.sqlite

train_X.head(2)
test_y.head()
    #check the datasets

#Support Vector Machine (SVM)
model = svm.SVC()
    #select the algorithm
model.fit(train_X,train_y)
    #train the algorithm with the training data and the training check_output
prediction = model.predict(test_X)
    #pass the test data to the trained algorithm`
print('The accurqacy of the SVM is: ', metrics.accuracy_score(prediction, test_y))
    #check the accuracy of the algorithm
    #we pass the predicted ouput by the model and the actual output

#Train different algorithms to see differences

# LOGISTIC REGRESSION
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the Logistic Regression is', metrics.accuracy_score(prediction,test_y))

#DECISION TREE
model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the Decision Tree is: ', metrics.accuracy_score(prediction,test_y))

#K-Nearest Neighbours
model = KNeighborsClassifier(n_neighbors=3)
    #this examines 3 neighbours for putting the new data into a class
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('The accuracy of the KNN is: ', metrics.accuracy_score(prediction,test_y))

#Now check accuracy for various n_neighbors in KNN
a_index = list(range(1,11))
a=pd.Series()
x=[1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X,train_y)
    prediction = model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y)))
plt.plot(a_index,a)
plt.xticks(x)
plt.show()

#Now try using ML algorithms using Petal and Sepal features seperately
petal = iris[['PetalLengthCm', 'PetalWidthCm', 'Species']]
sepal = iris[['SepalLengthCm', 'SepalWidthCm', 'Species']]

train_p,test_p = train_test_split(petal, test_size = 0.3, random_state=0)
    #petals
train_x_p = train_p[['PetalWidthCm', 'PetalLengthCm']]
train_y_p = train_p.Species
test_x_p = test_p[['PetalWidthCm', 'PetalLengthCm']]
test_y_p = test_p.Species

train_s, test_s = train_test_split(sepal, test_size = 0.3, random_state = 0)
    #Sepal
train_x_s = train_s[['SepalLengthCm','SepalWidthCm']]
train_y_s = train_s.Species
test_x_s = test[['SepalLengthCm','SepalWidthCm']]
test_y_s = test_s.Species

#SVM
#petal
model = svm.SVC()
model.fit(train_x_p, train_y_p)
prediction = model.predict(test_x_p)
print('Accuracy of the SVM using Petal is: ', metrics.accuracy_score(prediction,test_y_p))

#sepal
model = svm.SVC()
model.fit(train_x_s,train_y_s)
prediction = model.predict(test_x_s)
print('Accuracy of the SVM using Sepal is: ', metrics.accuracy_score(prediction, test_y_p))

#LOGISTIC REGRESSION
#Petal
model = LogisticRegression()
model.fit(train_x_p, train_y_p)
prediction = model.predict(test_x_p)
print('Accuracy of the Logistic Regression using Petal is: ', metrics.accuracy_score(prediction, test_y_p))

#Sepal
model = LogisticRegression()
model.fit(train_x_s,train_y_s)
prediction = model.predict(test_x_s)
print('Accuracy of Logistic Regression for Sepal is: ', metrics.accuracy_score(prediction, test_y_s))

#DECISION TREE
#petal
model = DecisionTreeClassifier()
model.fit(train_x_p, train_y_p)
prediction = model.predict(test_x_p)
print('Accuracy of the Decision Tree using Petal is: ', metrics.accuracy_score(prediction, test_y_p))

#Sepal
model = DecisionTreeClassifier()
model.fit(train_x_s,train_y_s)
prediction = model.predict(test_x_s)
print('Accuracy of Decision Tree for Sepal is: ', metrics.accuracy_score(prediction, test_y_s))

#KNN
#Petal
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_x_p, train_y_p)
prediction = model.predict(test_x_p)
print('Accuracy of the KNN using Petal is: ', metrics.accuracy_score(prediction, test_y_p))


#Sepal
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(train_x_s,train_y_s)
prediction = model.predict(test_x_s)
print('Accuracy of KNN for Sepal is: ', metrics.accuracy_score(prediction, test_y_s))

#Note that using the petal data (highly correlated) yields
#much more accurate results vs the Sepal data (low correlation)
# This makes sense and is noted in the heatmap above
