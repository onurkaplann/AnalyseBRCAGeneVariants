# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:38:29 2020

@author: Onur
"""

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

#Data insertion
data = pd.read_csv("variants_encoded.csv")

x = data.drop(['rcv.clinical_significance'], axis=1).to_numpy()
y = data['rcv.clinical_significance'].to_numpy()

x1 = data.drop(['rcv.clinical_significance'], axis=1)
y1 = data['rcv.clinical_significance']


#Selecting LightGBM İnstance
model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.05, max_delta_step=0, max_depth=10,
              min_child_weight=6, missing=None, n_estimators=200, n_jobs=1,
              nthread=None, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1) 

model2 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=10,
              min_child_weight=1, missing=None, n_estimators=200, n_jobs=1,
              nthread=None, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

#model = select_model(x,y,xgb.XGBClassifier()) #Enable this for run Grid-Search

#Selecting the Model Which Have Best Parameters
def select_model(x, y, model):
    params  = {"max_depth": [10,30,50],
              "min_child_weight" : [1,3,6],
              "n_estimators": [200],
              "learning_rate": [0.05, 0.1,0.16]}
    
    params2  = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9]}
     
    #grid_search = GridSearchCV(model, param_grid=params, cv = 3, verbose=10, n_jobs=-1)
    grid_search = GridSearchCV(model, param_grid=params, cv = 5, verbose=10, scoring="roc_auc_ovr_weighted", n_jobs=-1)
    grid_search.fit(x, y)
    print('Best Estimator:', grid_search.best_estimator_,'\n'+'Best Score:', grid_search.best_score_)
    return grid_search.best_estimator_

#variables for calculation of average classification reports
originalclass = []
predictedclass = []
accuracymean = []

#Average of all cross_validation fold's classification reports
def all_classification_reports(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    
#Model Training and Getting Performance from Variants of Known Clinical Significance
def main(x,y,model):
    skf = StratifiedKFold(n_splits=5, random_state=None)
    counter = 0
    for train_index, test_index in skf.split(x,y):
        train = x[train_index]
        test = x[test_index] 
    
        y_train = y[train_index]
        y_test = y[test_index]
    
        model.fit(train,y_train)         
        y_pred = model.predict(test)
                
        # accuracy = accuracy_score(y_pred, y_test)*100
        # print("XGBOOST: ",accuracy,"%")
        
        accuracymean.append(accuracy_score(y_pred, y_test)*100)
            
        # report = metrics.classification_report(y_pred, y_test)
        # print(report)       
        
        # print('part:'+str(counter),metrics.classification_report(y_pred, y_test))
        counter += 1     
        all_classification_reports(y_pred, y_test)
    
    # Average values in classification report for all folds in a K-fold Cross-validation    
    print("Ortalama Accuracy : ")
    avg_score = np.mean(accuracymean,axis=0)
    print(avg_score)
    
    print("Ortalama Classification Report : ")
    print(metrics.classification_report(originalclass, predictedclass)) 
        
    print("Parametreler :")
    print(model.get_params())
    print("Bitti")    
    print(model.get_xgb_params)
    
#Determination of Important Features in the Model
def feature_IMP(x,y,model):   
    train, test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.2)
    model.fit(train,y_train)
    #auc(model, train, test)
        
    y_pred = model.predict(test)
            
    accuracy = accuracy_score(y_pred, y_test)*100
    print("XGBOOST: ",accuracy,"%")
    
    report = metrics.classification_report(y_pred, y_test)
    print(report)       
    
    FT2 = model.get_booster().get_score(importance_type="gain")
    print(FT2)
    
    sorted_idx = np.argsort(model.feature_importances_)[::-1]
    for index in sorted_idx:
        print([train.columns[index], model.feature_importances_[index]]) 
        
    fig, ax = plt.subplots(figsize=(12,12))
    xgb.plot_importance(model, importance_type='gain', max_num_features=10,ax=ax)
    plt.show()

#Estimation Clinical Significance from Variants of Uncertain Significance
def estimete_VUS(x,y,params):    
    vus = pd.read_csv("variants_encoded_only_VUS.csv")
    vusx = vus.drop(["rcv.clinical_significance"], axis=1)
    vusy = vus["rcv.clinical_significance"]
    
    model.fit(x,y)
        
    y_pred = model.predict(vusx)
            
    accuracy = accuracy_score(y_pred, vusy)*100
    print("XGBOOST: ",accuracy,"%")
    
    report = metrics.classification_report(y_pred, vusy)
    print(report)       
    
    FT2 = model.get_booster().get_score(importance_type="gain")
    print(FT2)
    
    sorted_idx = np.argsort(model.feature_importances_)[::-1]
    for index in sorted_idx:
        print([x.columns[index], model.feature_importances_[index]]) 
    
    # y_pred = pd.DataFrame(y_pred,columns=["XGBoost"])
    # y_pred.to_csv('XGBoostVus.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(12,12))
    xgb.plot_importance(model, importance_type='gain', max_num_features=10, ax=ax,xlabel = "VUS")
    plt.show()



main(x,y,model)
feature_IMP(x1,y1,model)
estimete_VUS(x1,y1,model)