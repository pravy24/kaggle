# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:14:05 2017

@author: Z003EY2A
"""
import numpy as npy
import pandas as pd
#import matplotlib.pyplot as plot

    
""" Cost Function """
def computeCost(X, y, theta, lamda):
    
    cost = 0
    m = len(y)
    
    #errors = (X.dot(theta) - y ) **2
    #cost = (1/(2*m)) * sum(errors)
    
    # Cost function can also be written as:-
    # (1/2m) * T(X * theta) * (X * theta)
    cost = 1/(2*m) * (X.dot(theta) - y).T.dot(X.dot(theta) - y) + \
                                             lamda/(2*m) * sum(theta * theta)
    return (cost[0])



""" Gradient Descent algorithm """
def gradientDescent(X, y, theta, alpha, lamda, iterations):

    m = len(y)
    n = len(X.columns) #Get the number of features including the intercept term.
    cost_hist = npy.zeros((iterations, 1))
    theta_t = theta.copy(deep=True) # theta_t = theta is incorrect, as it will make only a reference

    i = 0
    while (i < iterations):
        cost_hist[i] = computeCost(X, y, theta, lamda)
        print(cost_hist[i])
        j = 0
        while j < n:
            theta_t.iloc[j:j+1,0:1] = \
                        theta_t.iloc[j,0] - \
                                    (alpha/m) * (X.dot(theta) - y).T.dot(X.iloc[0:m,j])[0] + \
                                    (lamda / m) * j * theta.iloc[j,0]
            j = j+1

        theta = theta_t.copy(deep=True)
        i = i + 1

    return ((theta, cost_hist))



def normalize_feature(X):
    
        meanX = X.mean(axis=0)
        meanX[0] = 0.0
        stdX = X.std(axis=0)
        stdX[0] = 1.0
        
        return((X - meanX)/stdX)
    
        

def predict(X, theta):
    
    return (X.dot(theta))




""" The MAIN """
if __name__ == "__main__":

    alpha = .1
    lamda = 1
    iterations = 1500
        
    # read_csv : load the csv file into a DataFrame,
    # usecols : list of column names to be loaded,
    # na_values : A dict of columns and a list of values to be interpreted as NaN,
    # keep_default_na : if False, then override the default NaN values.'''
    # fillna(0) : replaces the default NA values with 0,
    hpdata = pd.read_csv(\
                "..\data\hp_train.csv", 
                usecols=['LotFrontage', 'LotArea', 'YearBuilt',
                         'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 
                         'TotalBsmtSF', '1stFlrSF','2ndFlrSF', 'LowQualFinSF', 
                         'GrLivArea','BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                         'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                         'Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 
                         'OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 
                         'PoolArea','MiscVal', 'MoSold', 'YrSold', 'SalePrice'],
                na_values={'LotFrontage':['NA', 'na']},
                keep_default_na=False).fillna(0)

#    hpdata = pd.read_csv(\
#                "..\data\hp_train.csv", 
#                usecols=['LotFrontage', 'LotArea', 'SalePrice'],
#                na_values={'LotFrontage':['NA', 'na'], 'LotArea':['NA', 'na']},
#                keep_default_na=False).fillna(0)

    m = len(hpdata)
    n = len(hpdata.columns)-1 #exclude the predictor
    X = hpdata.iloc[0:m, 0:n] # DataFrame.iloc = index locator
    y = hpdata.iloc[0:m, n:n+1]
    y.columns = [0]
    
#    """numpy.ones creates an array of 1.0"""
#    """numpy.concatenate merges two arrays"""
#    """axis=1 for concatenate merges the arrays vertically"""
#    X = X.rename_axis({'LotFrontage':1, 'LotArea':2, 'YearBuilt':3, 'MasVnrArea':4, 
#                   'BsmtFinSF1':5, 'BsmtFinSF2':6,'BsmtUnfSF':7, 'TotalBsmtSF':8, 
#                   '1stFlrSF':9,'2ndFlrSF':10, 'LowQualFinSF':11, 'GrLivArea':12,
#                   'BsmtFullBath':13, 'BsmtHalfBath':14, 'FullBath':15, 'HalfBath':16, 
#                   'BedroomAbvGr':17, 'KitchenAbvGr':18, 'TotRmsAbvGrd':19, 'Fireplaces':20, 
#                   'GarageCars':21, 'GarageArea':22,'WoodDeckSF':23, 'OpenPorchSF':24, 
#                   'EnclosedPorch':25,'3SsnPorch':26, 'ScreenPorch':27, 'PoolArea':28,
#                   'MiscVal':29, 'MoSold':30, 'YrSold':31}, axis='columns')
    
    X = pd.concat([pd.DataFrame(npy.ones((m,1)), columns=['Intercept']), X], axis=1)
    n = n+1 #since an intercept column is added
    
    X_norm = normalize_feature(X)
    
    theta = pd.DataFrame(npy.zeros((n, 1)), index=['Intercept', 'LotFrontage', 'LotArea', 'YearBuilt',
                         'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 
                         'TotalBsmtSF', '1stFlrSF','2ndFlrSF', 'LowQualFinSF', 
                         'GrLivArea','BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                         'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                         'Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 
                         'OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 
                         'PoolArea','MiscVal', 'MoSold', 'YrSold'])
    
    cost = computeCost (X_norm, y, theta, lamda)

    (theta, cost_hist) = gradientDescent(X_norm, y, theta, alpha, lamda, iterations)
        
    #p = predict(npy.array([1, 3.5]), theta)
    print(theta)
    
    #print(p)