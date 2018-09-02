# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 09:14:33 2018

@author: Shariq
"""


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score


warnings.filterwarnings("ignore")

# Replaces 0 to 0.0001
def replaceZeroes(data):
    data[data == 0] = 10**-4
    return data

# Reading the dataset
#dataset = pd.read_csv("boston_house_data.csv")
#dataset = pd.read_csv("ailerons.csv")
dataset = pd.read_csv("forestFires.csv")

"""
###
dataset = pd.read_csv("abalone.csv")
dataset = pd.get_dummies(dataset)
cols = dataset.columns.tolist()
cols = cols[-3:] + cols[:-3]
dataset = dataset[cols]
###
"""

#dataset = pd.read_csv("airfoil_self_noise.txt", sep = "\t", header = None)

#dataset = pd.read_csv("Concrete_Data.csv", sep = "\t")

dataset.shape

#Normalizing the dataset ussing preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataset)

#Replace all 0 with a minimum value close to zero to resolve log(0) issue
x_scaled = replaceZeroes(x_scaled)
dataset = pd.DataFrame(x_scaled)

# Renaming the dataset columns 
# dataset.columns = ['X1','X2','X3','X4','X5','y']
XColsSize = dataset.shape[1] - 1
XColsName = ['X{}'.format(x+1) for x in range(0, XColsSize)]
FFXColsName = np.copy(XColsName)
XColsName.append('y')
XColsName

dataset.columns = XColsName

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

# R2 score on train data
print(X_train.shape, y_train.shape)

# R2 score on test data
print(X_test.shape, y_test.shape)

## FFX
import ffx
models = ffx.run(X_train, y_train, X_test, y_test, FFXColsName)

X_test_matrix = X_test.as_matrix()
print("FFX")
for model in models:
    y_pred = model.simulate(X_test_matrix)
    #print(r2_score(y_test, y_pred))
    #print(model)
print("Train:", r2_score(y_test, y_pred))
print("Test:", r2_score(y_test, y_pred))
print("Model:",model)
a = str(model)
print("Model Size:", np.char.count(a, '+') + np.char.count(a, '-') + 1)
