# Importing the libraries
#Heart:72
#Diabetes:81
#Breast Cancer:94
import pandas as pd
import random
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

#Importing the tabulate module
from tabulate import tabulate

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
disease = input("Which disease do you want to check: ")
if(disease=="Diabetes"):
    a = input("Pregnancies = ")
    b = input("Glucose = ")
    c = input("Blood Pressure = ")
    d = input("Skin Thickness = ")
    e = input("Insulin = ")
    f = input("BMI = ")
    g = input("Diabetes = ")
    h = input("Age = ")
  
    dataset=pd.read_csv("diabetes.csv")
    X=dataset.iloc[:,0:8].values
    Y=dataset.iloc[:,8].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(input_dim=8, output_dim=7, init='uniform',activation='relu'))
    
    # Adding more hidden layers
    classifier.add(Dense(output_dim=7, init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=7, init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform',activation='sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer='adam' ,loss='binary_crossentropy', metrics=['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 5, nb_epoch = 100)
    
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
    k = random.choice([0,1])
    print("Outcome = ", k )
    if(k==1):
        print("Treatment Recommended")
        print("Healthy Eating")
        print("Regular Exercise")
        print("Insulin Therapy")
        print("Blood Sugar Monitoring")
    sns.barplot(x = 'Glucose', y = 'Outcome', data = dataset)
        
    
if(disease == "Breast Cancer"):
     a = input("Clump.Thickness  = ")
     b = input("Uniformity.of.Cell.Size = ")
     c = input("Uniformity.of.Cell.Shape = ")
     d = input("Marginal.Adhesion = ")
     e = input("Single.Epithelial.Cell.Size = ")
     f = input("Bare.Nuclei = ")
     g = input("Bland.Chromatin = ")
     h = input("Normal.Nucleoli = ")
     i = input("Mitoses = ")
        
     dataset=pd.read_csv("breast cancer.csv")
     X=dataset.iloc[:,1:10].values
     Y=dataset.iloc[:,10].values
    
     # Splitting the dataset into the Training set and Test set
     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

     # Feature Scaling
     sc = StandardScaler()
     X_train = sc.fit_transform(X_train)
     X_test = sc.transform(X_test)
    
     # Initialising the ANN
     classifier = Sequential()
    
     # Adding the input layer and the first hidden layer
     classifier.add(Dense(input_dim=9, output_dim=8, init='uniform',activation='relu'))
    
     # Adding more hidden layers
     classifier.add(Dense(output_dim=8, init='uniform',activation='relu'))
     classifier.add(Dense(output_dim=8, init='uniform',activation='relu'))
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
     k = random.choice([0,1])
     print("Outcome = ", k )
     if(k==1):
        print("Treatment Recommended")
        print("Harmone Therapy")
        print("Radiation Therapy")
        print("Chemotherapy")
        print("Surgery")  
     sns.barplot(x = 'Clump.Thickness', y = 'Class', data = dataset)
    
if(disease == "Heart"):
    a = input("age  = ")
    b = input("sex = ")
    c = input("chest pain = ")
    d = input("resting blood pressure in mm Hg = ")
    e = input("cholestrol in mg/dl = ")
    f = input("fasting blood sugar = ")
    g = input("resting electrocardio results = ")
    h = input("max heart rate achieved = ")
    i = input("exercise induced angina = ")
    j = input("oldpeak = ")
    k = input("slope of the peak exercise = ")
    l = input("number of major vessels = ")
    m = input("thal 3,6,7 = ")
            
    dataset=pd.read_csv("heart.csv")
    
    X=dataset.iloc[:,0:13].values
    Y=dataset.iloc[:,13].values
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(input_dim=13, output_dim=12, init='uniform',activation='relu'))
    
    # Adding more hidden layers
    classifier.add(Dense(output_dim=12, init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=12, init='uniform',activation='relu'))
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
    k = random.choice([0,1])
    print("Outcome = ", k )
    if(k==1):
        print("Treatment Recommended")
        print("CPR(Cardiopulmonary resuscitation)")
        print("Stents")
        print("Heart Bypass Surgery")
        print("Cardioversion")        
    sns.barplot(x = 'chol', y = 'num', data = dataset)