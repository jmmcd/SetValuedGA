# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 23:05:18 2018

@author: Shariq
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import ElasticNetCV
from operator import itemgetter
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# Replaces 0 to 0.0001
def replaceZeroes(data):
    data[data == 0] = 10**-4
    return data

# Reading the dataset
#dataset = pd.read_csv("boston_house_data.csv")
#dataset = pd.read_csv("ailerons.csv")
#dataset = pd.read_csv("forestFires.csv")

"""
dataset = pd.read_csv("abalone.csv")
dataset = pd.get_dummies(dataset)
cols = dataset.columns.tolist()
cols = cols[-3:] + cols[:-3]
dataset = dataset[cols]
"""

#dataset = pd.read_csv("airfoil_self_noise.txt", sep = "\t", header = None)

dataset = pd.read_csv("Concrete_Data.csv", sep = "\t")

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
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Elastic net without GA
regr = ElasticNetCV(random_state=0, cv = 5)
regr.fit(X_train,y_train)
print("Elastic Net:")
# R2 score on train data
print("Train:",regr.score(X_train,y_train))

#print(regr.score(X_test, y_test))

# Sort the coeficients
def sortCoef(columns, coef):
    nlist = [(y, x) for x,y in zip(columns, coef)]
    try:
        nlist = sorted(nlist, key=itemgetter(0), reverse = True)
    except ValueError:
        print("Error nlist:", nlist)
    return [val for (key, val) in nlist], [key for (key, val) in nlist]

# print a number to 3 significant digits
def coefStr(x):
    if x == 0.0:
        s = '0'
    elif np.abs(x) < 1e-4: s = ('%.2e' % x).replace('e-0', 'e-')
    elif np.abs(x) < 1e-3: s = '%.6f' % x
    elif np.abs(x) < 1e-2: s = '%.5f' % x
    elif np.abs(x) < 1e-1: s = '%.4f' % x
    elif np.abs(x) < 1e0:  s = '%.3f' % x
    elif np.abs(x) < 1e1:  s = '%.2f' % x
    elif np.abs(x) < 1e2:  s = '%.1f' % x
    elif np.abs(x) < 1e4:  s = '%.0f' % x
    else:                     s = ('%.2e' % x).replace('e+0', 'e')
    return s

y_pred = regr.predict(X_test)
# R2 score on test data
print("Test:",r2_score(y_test, y_pred))
columns, regr.coef_ = sortCoef(X.columns, regr.coef_)
model = ""
i=0

# Create the model equation 
if regr.intercept_ not in [0,-0]:
    model = str(coefStr(regr.intercept_))
for ind in columns:
    if regr.coef_[i] not in [0,-0]: 
        if "-" in str(regr.coef_[i]): 
            indCoef = str(coefStr(regr.coef_[i]))+"*"+str(ind) 
        elif len(model) > 0:   
            indCoef = "+" + str(coefStr(regr.coef_[i]))+"*"+ ind
        else:
            indCoef = str(coefStr(regr.coef_[i]))+"*"+ ind
        model = model + indCoef
    i = i + 1
print("Model:", model)
print("Model size:", np.count_nonzero(regr.coef_)+1)
