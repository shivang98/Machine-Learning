# Data Preprocessing

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Handle missing data -> replace nan data with mean or other strategies
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# encoding categorical data - because our algo didnt understand words written so 
# we use labelencoder to encode them to numerics
# we use onehotencoder because python can think that our features are not categorical i.e
# 0 < 1 < 2 but actually the doesnt have any relation between them
# so we apply this to create different matrix to solve the problem    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# cross validation -> to seperate training and testing data to check accuracy of our model
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)

# feature scaling -> most of the algorithms are based on eculidian distance formula 
# so if we dont apply feature scaling the larger feature will dominate other features
# and we dont get correct output
# also if we do feature scalling we will converge faster
# feature scalling is of two types
# standard (x-mean)/sd and normalise (x-min)/range 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
