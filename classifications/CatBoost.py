# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:12:26 2020

@author: EmreKARA
"""

import catboost as cb
from catboost.utils import get_roc_curve
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


## Data insertion
VUS= pd.read_csv("variants_not_encoded_only_VUS.csv")
data= pd.read_csv("variants_not_encoded.csv")
x = data.drop(['rcv.clinical_significance'], axis=1).to_numpy()
y = data['rcv.clinical_significance'].to_numpy()

#Get Categorical features
cat_features_index = [data.columns.get_loc(col) for col in data if data[col].dtypes.name in ['object','bool']] #Categoric Feature'ların indexini tutan dizi.
cat_features_index = cat_features_index[:-1] #Target sınıf çıkarılır

#Selecting CatBoost Instance
cb_classifier = cb.CatBoostClassifier(bagging_temperature=0.3, depth=7, grow_policy='Lossguide', iterations= 1, l2_leaf_reg= 1, learning_rate= 0.1, silent=True)
# cb_classifier = cb.CatBoostClassifier(eval_metric="AUC",one_hot_max_size=31, depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15, silent=True)
# cb_classifier = select_model(x, y, cb.CatBoostClassifier(), cat_features=cat_features_index)

#variables for calculation of average classification reports
originalclass = []
predictedclass = []
accuracies = []

#Average of all cross_validation fold's classification reports
def all_classification_reports(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)

#Selecting the Model Which Have Best Parameters
def select_model(x, y, model, cat_features=None):
    params = {'depth': [4, 7, 10],
              'learning_rate' : [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1,4,9],
            'iterations': [200],
            'bagging_temperature':[0.3 , 1, 10],
            'grow_policy':['SymmetricTree', 'Depthwise', 'Lossguide']}
    params_old = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1,4,9],
         'iterations': [500]} 
    grid_search = GridSearchCV(model, params, scoring ='roc_auc_ovr_weighted', cv = 5, n_jobs=-1) # set "params" to "params_old" for change parameter set
    grid_search.fit(x, y, cat_features = cat_features)
    print('Best Params:', grid_search.best_params_,'\n'+'Best Score:',grid_search.best_score_)
    return grid_search.best_params_
    
#Plotting Feature Importance Table of CatBoost
def plot_feature_importance(data):
    x_train, x_test, y_train, y_test = train_test_split(data.drop(['rcv.clinical_significance'], axis=1), data['rcv.clinical_significance'],random_state=10, test_size=0.2)
    cb_classifier = cb.CatBoostClassifier(bagging_temperature=0.3, depth=7, grow_policy='Lossguide', iterations= 500, l2_leaf_reg= 1, learning_rate= 0.1, silent=True)
    cb_classifier = cb.CatBoostClassifier(eval_metric="AUC",one_hot_max_size=31, depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15, silent=True)
    cb_classifier.fit(x_train,y_train, cat_features= cat_features_index)
    model = cb_classifier
    
    feature_importance_coef_all = model.get_feature_importance().tolist()
    feature_importance_coef_selected_index = sorted(range(len(feature_importance_coef_all)),key=feature_importance_coef_all.__getitem__, reverse=True)[:10]
    feature_names = [data.columns[i] for i in feature_importance_coef_selected_index]
    feature_importance_degrees = [feature_importance_coef_all[i] for i in feature_importance_coef_selected_index]
    feature_importance = [feature_names, feature_importance_degrees]
    
    plt.rcdefaults()
    fig = plt.figure('Feature Importance Degree From CatBoost', figsize=(14,22))
    ax = fig.add_subplot(1,1,1)
    
    y_pos = np.arange(len(feature_importance[0]))
    ax.barh(y_pos, feature_importance[1], align='center', height=0.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_importance[0])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance')
    
    for i, v in enumerate(feature_importance[1]):
        ax.text(v, i, " {:.2f}".format(v), color='black', va='center')
    plt.grid(alpha=0.5)
    plt.show()
    plt.savefig('catboost_feature_importance.png')

#Estimation Clinical Significance from Variants of Uncertain Significance 
def estimate_VUS(VUS,x_train, y_train, cat_features_index, model):
    x_test = VUS.drop(['rcv.clinical_significance'], axis=1).to_numpy()
    y_test = VUS['rcv.clinical_significance'].to_numpy()
    cb_classifier.fit(x_train,y_train, cat_features= cat_features_index)
    y_pred = cb_classifier.predict(x_test)
    y_pred = y_pred.tolist()
    vus_predictions = pd.DataFrame(y_pred,columns=['CatBoost'])
    vus_predictions.to_csv('catboost_vus_predictions.csv',index=False)
    print(metrics.classification_report(y_pred, y_test))

#Model Training and Getting Performance from Variants of Known Clinical Significance
def main(data,model):
    target_names = ['Benign', 'Likely beging', 'Likely pathogenic', 'Pathegenic']
    skf = StratifiedKFold(n_splits=5)
    counter = 0
    for train_index, test_index in skf.split(x,y):
        x_train = x[train_index]
        x_test = x[test_index] 
        
        y_train = y[train_index]
        y_test = y[test_index]    
        
        model.fit(x_train,y_train, cat_features= cat_features_index)
        
        y_pred = model.predict(x_test)
        y_pred = y_pred.tolist()
        print('part:'+str(counter),metrics.classification_report(y_test, y_pred))
        counter += 1
        
        accuracies.append(metrics.accuracy_score(y_test, y_pred))
        all_classification_reports(y_test, y_pred)
       
    print('average accuracy:', np.mean(accuracies,axis=0))
    print('average classification report:\n'+metrics.classification_report(originalclass, predictedclass, target_names = target_names))



plot_feature_importance(data)
main(model=cb_classifier, data=data)
estimate_VUS(VUS, x, y, cat_features_index, cb_classifier)





    


