# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:09:51 2020

@author: EmreKARA
"""
import pandas as pd
import numpy as np
from copy import deepcopy
class Preprocess:
    def __init__(self, data_path_c=None,read_dtype=None,autoImpute=True, autoEliminateNullColumns=True):
        try:
            self.data=pd.read_csv(data_path_c, dtype=read_dtype)
            self.source_columns = None
            self.target_columns = None
            if autoEliminateNullColumns:
                self.eliminateNullColumns()
            if autoImpute:
                self.impute()
        except:
            print('Can not initialize')
    def print(self,cols=None):
        try:
            if cols == None:
                cols = self.data
            print(self.data)
        except:
            print('Can not print')
    def impute(self,cols=None,strategy_n='mean',strategy_s='most_frequent',fill_value_c = None):
        try:
            if (not self.data.isnull().sum().any()):
                print('Null Values Not Found')
                return
            if(cols == None):
                cols=self.data
            else:
                cols = self.getCols(cols)
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(missing_values=np.NaN , strategy=strategy_n)
            imputer_s = SimpleImputer(missing_values=np.NaN , strategy= strategy_s, fill_value=fill_value_c)
            for column_name in cols:
                if self.data[column_name].isnull().values.any():
                    column = self.data[[column_name]]
                    try:
                        column_temp = imputer.fit_transform(column)
                    except (AttributeError,ValueError):
                        column_temp = imputer_s.fit_transform(column)
    #                except Exception as ex:
    #                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    #                    message = template.format(type(ex).__name__, ex.args)
    #                    print(message)
                    self.__updateColumn(column_temp,column_name)
        except Exception as exc:
            print('Can not impute', exc)
    def dropCols(self,cols):
        try:
            cols = self.getCols(cols)
            self.data.drop(columns=cols,  inplace=True)
            return self.data
        except:
            print('Can not Drop Columns')
    def encode(self,cols=None,encoder = 'LabelEncoder'):
        try:
            if(cols == None):
                 acceptable_dtypes = ['int64','float64']
                 cat_features_name = [col for col in self.data if self.data[col].dtypes.name not in acceptable_dtypes] #Categoric Feature'ların adlarını tutan dizi.
                 cat_features_name = cat_features_name[:-1] #Target sınıf çıkarılır
                 cols = self.getCols(cat_features_name)
            else:
                cols = self.getCols(cols)
            if encoder == 'LabelEncoder':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for column_name in cols:
                    column = self.data[column_name].values.ravel()
                    column_temp = le.fit_transform(column)
                    self.__updateColumn(column_temp, column_name)
                    self.data[column_name] = self.data[column_name].astype(np.float64)
            elif encoder == 'OneHotEncoder':
                print('Not implemented')
        except Exception as ex:
            print('Can not Encode')
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
    def __updateColumn(self,arr,arr_name):
        new_column = pd.Series(arr.ravel(), name=arr_name)
        self.data.update(new_column)
    def trainTestSplitting(self,target_columns,source_columns=None, test_size_c=0.33, random_state_c=0):
        try:
            from sklearn.model_selection import train_test_split
            if source_columns == None:
                target_columns = self.getCols(target_columns)
                source_columns = self.data
                for i in target_columns:
                    source_columns = source_columns.drop(columns = i)
            else:
                source_columns = self.getCols(source_columns)
                target_columns = self.getCols(target_columns)
            x_train, x_test, y_train, y_test = train_test_split(source_columns, target_columns, test_size = test_size_c, random_state = random_state_c)
            return x_train, x_test, y_train, y_test
        except:
            print('Can not Split for Train-Test')
            return [],[],[],[]
    def scale(self,cols=None, onData=True):
        try:
            if(cols == None):
                cols=self.data
            else:
                cols = self.getCols(cols)
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            for column_name in cols:
                column = self.data[[column_name]].astype(float)
                column = sc.fit_transform(column)
                self.__updateColumn(column, column_name)
        except Exception as e:
            print('Can not Scale')
            print(e)
    def bWElimination(self,target_columns,source_columns=None,pValue=0.05):
        import statsmodels.api as sm
        import numpy as np
        if source_columns == None:
            if source_columns == None:
                target_columns = self.getCols(target_columns)
                source_columns = self.data
                for i in target_columns:
                    source_columns = source_columns.drop(columns = i)
        else:
            source_columns = self.getCols(source_columns)
            target_columns = self.getCols(target_columns)
        be_list = sm.add_constant(source_columns.to_numpy()) #fit_intercept.py
        while(True):
            results = sm.OLS(endog = target_columns.astype(float), exog= be_list.astype(float)).fit()
            p_valuesArray = []
            for j in range(results.params.size):
                r_temp = np.zeros_like(results.params)
                r_temp[j] = 1
                T_test = results.t_test(r_temp)
                p_value = T_test.pvalue.item(0)
                p_valuesArray.append(float(p_value))
            maxPValue = max(p_valuesArray)
            if maxPValue > pValue:
                source_columns = source_columns.drop(source_columns.columns[(p_valuesArray.index(max(p_valuesArray)))-1], axis='columns')#-1 :> numpy array ve Dataframe de const dan dolayı index farkı oluşur
                be_list = np.delete(arr=be_list,obj=p_valuesArray.index(max(p_valuesArray)) ,axis=1)
            else:
                self.data = pd.concat([source_columns,target_columns], axis=1)
                break
    def getCols(self, cols):
        if all(isinstance(n, int) for n in cols):
            cols_temp=[]
            for i in cols:
                cols_temp.append(self.data.columns[i])     
            cols = self.data[cols_temp]
            return cols
        if all(isinstance(n, str) for n in cols):
            return self.data[cols]
    def dropRows(self,rows):
        self.data.drop(rows, inplace = True)
    def eliminateNullColumns(self,percentage=0.20):
        self.data = self.data.loc[:, self.data.isnull().mean() < percentage]
    def getData(self):
        return deepcopy(self.data)
    def setData(self, data):
        del self.data
        self.data = data