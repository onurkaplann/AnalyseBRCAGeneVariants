# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:37:45 2020

@author: Onur
"""

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#Data insertion
data = pd.read_csv("variants_not_encoded.csv")

#Determination of Categorical Features
acceptable_dtypes = ['object','bool']
cat_features_name = [data.columns.get_loc(col) for col in data if data[col].dtypes.name in acceptable_dtypes] #The array that holds the index of the Categoric Features.
cat_features_name = cat_features_name[:-1] #Target class is subtracted

#Conversion of Categorical Features into Numerical
le = preprocessing.LabelEncoder()

num_of_columns = data.shape[1]

for i in range(0, num_of_columns):
    column_name = data.columns[i]
    column_type = data[column_name].dtypes
    
   
    if column_type == 'object' or  column_type == 'bool':
        #encode with sklearn
        le.fit(data[column_name])
        feature_classes = list(le.classes_)
        #print(feature_classes)
        
        encoded_feature = le.transform(data[column_name])
        data[column_name] = pd.DataFrame(encoded_feature)

x = data.drop(['rcv.clinical_significance'], axis=1).to_numpy()
y = data['rcv.clinical_significance'].to_numpy()

x1 = data.drop(["rcv.clinical_significance"], axis=1)
y1 = data["rcv.clinical_significance"]

#Selecting LightGBM İnstance
params={}
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['num_class']=4 #no.of unique values in the target class not inclusive of the end value
params['class_weight']='None' 
params['colsample_bytree']= 0.7
params['importance_type']= 'split'
params['learning_rate']=0.1
params['max_depth']=15
params['min_child_samples']=20
params['min_child_weight']=0.001
params['min_split_gain']=0.4
params['n_estimators']=400
params['n_jobs']=-1
params['num_leaves']=50
params['reg_alpha']=1.2
params['reg_lambda']=1.1
params['silent']='False'
params['subsample']=0.8
params['subsample_for_bin']=200000
params['subsample_freq']=20

params2={}
params2['boosting_type']='gbdt' #GradientBoostingDecisionTree
params2['objective']='multiclass' #Multi-class target feature
params2['metric']='multi_logloss' #metric for multi-class
params2['num_class']=4 #no.of unique values in the target class not inclusive of the end value
params2['learning_rate']=0.1
params2['max_depth']=15
params2['num_leaves']=100
params2['n_estimators']=400
params2['colsample_bytree']=0.7
params2['reg_alpha']=1.3
params2['reg_lambda']=1.2 
params2['min_split_gain']=0.4
params2['subsample']=0.9
params2['subsample_freq']=20

#params = select_model(x,y,lgb.LGBMClassifier()) #Enable this for run Grid-Search

target_names = ['Benign', 'Likely beging', 'Likely pathogenic', 'Pathegenic']
target_names2 = ['Benign', 'Likely benign', 'Likely pathogenic', 'Pathogenic','Uncertain Significance']

#Selecting the Model Which Have Best Parameters
def select_model(x, y, model):
    params = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'num_leaves': [50, 100, 200],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
    }
    
    params2 = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }
    
    
    #grid_search = GridSearchCV(model, n_jobs=-1, param_grid=params, cv = 3, scoring="roc_auc", verbose=5)
    grid_search = GridSearchCV(model, params, scoring ='roc_auc_ovr_weighted', cv = 5, n_jobs=-1)
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
def main(x,y,params):
    skf = StratifiedKFold(n_splits=5, random_state=None)
    counter = 0
    for train_index, test_index in skf.split(x,y):
        train = x[train_index]
        test = x[test_index] 
    
        y_train = y[train_index]
        y_test = y[test_index]
    
        d_train = lgb.Dataset(train, label=y_train)
        model = lgb.train(params, d_train, categorical_feature = cat_features_name)
        
        y_pred = model.predict(test)
    
        predictions_classes = []
        for i in y_pred:
            predictions_classes.append(np.argmax(i))
        
        predictions_classes = np.array(predictions_classes)
        
        #accuracy = accuracy_score(predictions_classes, y_test)*100
        #print("LIGHTGBM : ",accuracy,"%")
        
        accuracymean.append(accuracy_score(predictions_classes, y_test)*100)
        
        # report = metrics.classification_report(predictions_classes,y_test)
        # print(report)
        
        #print('part:'+str(counter),metrics.classification_report(predictions_classes, y_test, target_names=target_names))
        counter += 1
        all_classification_reports(y_test, predictions_classes)
        
    # Average values in classification report for all folds in a K-fold Cross-validation      
    print("Ortalama Accuracy : ")
    avg_score = np.mean(accuracymean,axis=0)
    print(avg_score)
    
    print("Ortalama Classification Report : ")
    print(metrics.classification_report(originalclass, predictedclass, target_names=target_names)) 
        
    
    print("Parametreler :")
    print(model.params)
    print("Bitti")    

#Determination of Important Features in the Model
def feature_IMP(x,y,params): 
    train, test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.2)   
    
    d_train = lgb.Dataset(train, label=y_train)
    model = lgb.train(params, d_train, categorical_feature = cat_features_name)
    
    y_pred = model.predict(test)

    predictions_classes = []
    for i in y_pred:
        predictions_classes.append(np.argmax(i))
    
    predictions_classes = np.array(predictions_classes)
    
    accuracy = accuracy_score(predictions_classes, y_test)*100
    print("LIGHTGBM : ",accuracy,"%")
    
    report = metrics.classification_report(predictions_classes,y_test)
    print(report)
    
    sorted_idx = np.argsort(model.feature_importance(importance_type='gain'))[::-1]
    for index in sorted_idx:
        print([train.columns[index], model.feature_importance(importance_type='gain')[index]]) 
       
    lgb.plot_importance(model, importance_type='gain', max_num_features=10,figsize=(12,12))
    plt.show()

    
#Estimation Clinical Significance from Variants of Uncertain Significance
def estimete_VUS(x,y,params):     
    vus = pd.read_csv("variants_encoded_only_VUS.csv")

    vusx = vus.drop(["rcv.clinical_significance"], axis=1)
    vusy = vus[["rcv.clinical_significance"]]
    
    vusy["rcv.clinical_significance"] = vusy["rcv.clinical_significance"].astype("category").cat.codes +4
    
    d_train = lgb.Dataset(x, label=y)
    
    model = lgb.train(params, d_train, categorical_feature = cat_features_name)
    
    y_pred = model.predict(vusx)

    predictions_classes = []
    for i in y_pred:
        predictions_classes.append(np.argmax(i))
    
    predictions_classes = np.array(predictions_classes)
    
    from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
    accuracy = accuracy_score(predictions_classes, vusy)*100
    print("LIGHTGBM : ",accuracy,"%")
    
    report = metrics.classification_report(predictions_classes,vusy,target_names = target_names2)
    print(report)
    
    sorted_idx = np.argsort(model.feature_importance(importance_type='gain'))[::-1]
    for index in sorted_idx:
        print([x.columns[index], model.feature_importance(importance_type='gain')[index]]) 
    
    le.fit(["Benign", "Likely benign", "Likely pathogenic", "Pathogenic"])
    predictions_classes = le.inverse_transform(predictions_classes)
    
    lgb.plot_importance(model, importance_type='gain',max_num_features=10,figsize=(12,12),xlabel = "VUS")
    plt.show()
    predictions_classes = pd.DataFrame(predictions_classes,columns=["LightGBM"])
    predictions_classes.to_csv('LightVus.csv', index=False)

main(x,y,params)    
feature_IMP(x1,y1,params)
estimete_VUS(x1,y1,params)
    
    
    