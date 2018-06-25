# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:14:56 2018

@author: Lenovo
"""

import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,usecols=[0,1,2])
df.columns=['Class label', 'Alcohol', 'Malic acid']

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()
scaled_df=scaler.fit_transform(df)


scale = MinMaxScaler()
scaled_mm = scale.fit_transform(df)


column_name=pd.DataFrame(scaled_mm, columns=['Class label', 'Alcohol', 'Malic acid'])