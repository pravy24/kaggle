# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:14:05 2017

@author: Z003EY2A
"""
import numpy as npy
import pandas as pd
from sklearn import linear_model as lr
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



def normalize_feature(X):
    
        meanX = X.mean(axis=0)
        meanX[0] = 0.0
        stdX = X.std(axis=0)
        stdX[0] = 1.0
        
        return((X - meanX)/stdX)

    
""" The MAIN """
if __name__ == "__main__":

        
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

    X = pd.concat([pd.DataFrame(npy.ones((m,1)), columns=['Intercept']), X], axis=1)
    n = n+1 #since an intercept column is added
    
    X_norm = normalize_feature(X)

    regr = lr.LinearRegression(fit_intercept=False)
    regr.fit(X_norm, y)

    theta = pd.DataFrame(regr.coef_.T, index=['Intercept', 'LotFrontage', 'LotArea',
                         'YearBuilt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 
                         'TotalBsmtSF', '1stFlrSF','2ndFlrSF', 'LowQualFinSF', 
                         'GrLivArea','BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                         'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                         'Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 
                         'OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 
                         'PoolArea','MiscVal', 'MoSold', 'YrSold'])
    print(theta)
    
    hptestdata = pd.read_csv(\
                "..\data\hp_test.csv", 
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

    m_test = len(hptestdata)
    n_test = len(hptestdata.columns)-1 #exclude the predictor
    X_test = hptestdata.iloc[0:m_test, 0:n_test] # DataFrame.iloc = index locator
    y_test = hptestdata.iloc[0:m_test, n_test:n_test+1]
    y_test.columns = [0]
    
    X_test = pd.concat([pd.DataFrame(npy.ones((m_test,1)), columns=['Intercept']), X_test], axis=1)
    n_test = n_test+1 #since an intercept column is added

    X_test_norm = normalize_feature(X_test)


    cost_test = computeCost (X_test_norm, y_test, theta, 0)
    
    #print(p)