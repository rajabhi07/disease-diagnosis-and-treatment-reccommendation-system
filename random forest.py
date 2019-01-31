
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('breast cancer.csv')
X=dataset.iloc[:,1:10].values
Y=dataset.iloc[:,10].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from tabulate import tabulate

accu = ((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))*100
precision=(cm[1][1]/(cm[0][1]+cm[1][1]))
recall=(cm[1][1]/(cm[1][0]+cm[1][1]))
f1=(2*precision*recall)/(precision + recall)

print(tabulate([['Accuracy',accu],['Precision',precision],
				['Recall',recall],['F1 Score',f1]],headers=['Factor','Value']))
