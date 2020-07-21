# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 18:39:29 2020

@author: Onur
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


#Data insertion
data = pd.read_csv('variants_encoded.csv')
x = data.drop(['rcv.clinical_significance'], axis=1).to_numpy()
y = data['rcv.clinical_significance'].to_numpy()

#Selecting LogisticRegression İnstance
logr = LogisticRegression(random_state = 0)
#logr = LogisticRegression(C=0.09, max_iter=50) # İnstance with selected_params by select_model() method
#logr = select_model(x,y,LogisticRegression()) #Enable this for run Grid-Search

#variables for calculation of average classification reports
originalclass = []
predictedclass = []
accuracymean = []

#Selecting the Model Which Have Best Parameters
def select_model(x, y, model):
    params = {'solver': ['newton-cg', 'saga', 'lbfgs'],
    'C':[0.01,0.09,0.5,1,5,10],
    'class_weight': ['balanced', None],
    'max_iter': [50,100,250,500]}
    
    grid_search = GridSearchCV(model, params, scoring ='roc_auc_ovr_weighted', cv = 5, n_jobs=-1)
    grid_search.fit(x, y)
    print('Best Estimator:', grid_search.best_estimator_,'\n'+'Best Score:', grid_search.best_score_)
    return grid_search.best_estimator_

#Average of all cross_validation fold's classification reports
def all_classification_reports(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)

#Model Training and Getting Performance from Variants of Known Clinical Significance
def main(x,y,logr):
    skf = StratifiedKFold(n_splits=5, random_state=None)
    counter = 0
    for train_index, test_index in skf.split(x,y):
        x_train = x[train_index]
        x_test = x[test_index] 
    
        y_train = y[train_index]
        y_test = y[test_index]
            
        sc = StandardScaler()
        
        X_train = sc.fit_transform(x_train)
        X_test = sc.transform(x_test)

        logr.fit(X_train,y_train)
        
        y_pred = logr.predict(X_test)
        
        # accuracy = accuracy_score(y_pred, y_test)*100
        # print("LogisticRegression : ",accuracy,"%")
        
        accuracymean.append(accuracy_score(y_pred, y_test)*100)
        
        # print('part:'+str(counter),metrics.classification_report(y_pred, y_test))
        
        all_classification_reports(y_test, y_pred)
        counter += 1
     
    # Average values in classification report for all folds in a K-fold Cross-validation  
    print("Average Accuracy : ")
    avg_score = np.mean(accuracymean,axis=0)
    print(avg_score)
    
    print("Average Classification Report : ")
    print(metrics.classification_report(originalclass, predictedclass)) 
    
    print("Parameters :")
    print(logr.get_params())
    print("END")
    

#Estimation Clinical Significance from Variants of Uncertain Significance
def estimete_VUS(x,y,logr):
    #Vus Data
    vus = pd.read_csv("variants_encoded_only_VUS.csv")
    vusx = vus.drop(["rcv.clinical_significance"], axis=1)
    vusy = vus["rcv.clinical_significance"]
    
    sc = StandardScaler()
    
    X_train = sc.fit_transform(x)
    X_test = sc.transform(vusx)
    
    logr.fit(X_train,y)
    
    y_pred = logr.predict(X_test)
    
    y_pred = pd.DataFrame(y_pred,columns=["LogisticRegression"])
    y_pred.to_csv('LogisticRegressionVUS.csv', index=False)
    
    accuracy = accuracy_score(y_pred, vusy)*100
    print("LogisticRegression Accuracy : ",accuracy,"%")
    
    print(metrics.classification_report(y_pred, vusy))
 

main(x,y,logr)    
estimete_VUS(x,y,logr)
