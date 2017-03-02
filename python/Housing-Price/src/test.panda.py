# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 21:15:04 2017

@author: Z003EY2A
"""

import pandas as pd

""" The MAIN """
if __name__ == "__main__":
        
    #data = npy.loadtxt("D:\IC019134\Learnings\Data Science\Coursera\Machine Learning - Andrew Ng\Excersise\Ex1\ex1data1.csv", delimiter=',')

    hpdata = pd.read_csv("D:\IC019134\Learnings\Data Science\workarea\python\Housing-Price\data\hp_train.csv", 
                         usecols=['Id', 'LotFrontage']).fillna(0)

    
    hpdata[0:3]