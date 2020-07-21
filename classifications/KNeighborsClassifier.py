# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 18:46:39 2020

@author: Onur
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

#Data insertion
data = pd.read_csv('variants_encoded.csv')

x = data.drop(['rcv.clinical_significance'], axis=1).to_numpy()
y = data['rcv.clinical_significance'].to_numpy()

#variables for calculation of average classification reports
originalclass = []
predictedclass = []
accuracymean = []

#Selected KNeighborsClassifier İnstance
knn = KNeighborsClassifier()

#Average of all cross_validation fold's classification reports
def all_classification_reports(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    
#Model Training and Getting Performance from Variants of Known Clinical Significance
def main(x,y,knn):
    skf = StratifiedKFold(n_splits=5, random_state=None)
    counter = 0
    for train_index, test_index in skf.split(x,y):
        x_train = x[train_index]
        x_test = x[test_index] 
    
        y_train = y[train_index]
        y_test = y[test_index]
        
        knn.fit(x_train,y_train)
        
        y_pred = knn.predict(x_test)
      
        # accuracy = accuracy_score(y_pred, y_test)*100
        # print("KNeighborsClassifier : ",accuracy,"%")
        
        accuracymean.append(accuracy_score(y_pred, y_test)*100)
        
        # print('part:'+str(counter),metrics.classification_report(y_pred, y_test))
        
        all_classification_reports(y_test, y_pred)
        counter += 1
        
    # Average values in classification report for all folds in a K-fold Cross-validation     
    print("Ortalama Accuracy : ")
    avg_score = np.mean(accuracymean,axis=0)
    print(avg_score)
    
    print("Ortalama Classification Report : ")
    print(metrics.classification_report(originalclass, predictedclass)) 
         
         
    print("Parametreler :")
    print(knn.get_params())
    print("Bitti")  

main(x,y,knn)
