# Importing the libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Importing the tabulate module
from tabulate import tabulate

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

dataset=pd.read_csv("lung cancer data sets.csv")

X=dataset.iloc[:,3:24].values
Y=dataset.iloc[:,24].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(input_dim=21, output_dim=20, init='uniform',activation='relu'))

# Adding more hidden layers
classifier.add(Dense(output_dim=20, init='uniform',activation='relu'))
classifier.add(Dense(output_dim=20, init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam' ,loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 5, nb_epoch = 200)

# Predicting the Test set results
y_pred=classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix is:\n",cm)

accu = ((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))*100
precision=(cm[1][1]/(cm[0][1]+cm[1][1]))
recall=(cm[1][1]/(cm[1][0]+cm[1][1]))
f1=(2*precision*recall)/(precision + recall)

print(tabulate([['Accuracy',accu],['Precision',precision],
				['Recall',recall],['F1 Score',f1]],headers=['Factor','Value']))


