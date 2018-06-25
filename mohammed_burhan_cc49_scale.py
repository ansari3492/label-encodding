# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:04:17 2018

@author: Lenovo
"""
import pandas as pd
data=pd.read_csv("Red_Wine.csv")

for i in data:
    data[i] = data[i].fillna(data[i].mode()[0])

features=data.iloc[:,0:-1].values
label=data["quality"].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
features[:,0]=le.fit_transform(features[:,0])

ohe = OneHotEncoder(categorical_features=[0])
features = ohe.fit_transform(features).toarray()

from sklearn.model_selection import train_test_split
features_train,features_test,label_train,label_test = train_test_split(features,label,test_size=0.2,random_state=0)

view_features=pd.DataFrame(features)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)