# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:12:26 2020

@author: EmreKARA
"""
import preprocessing
import numpy as np
from copy import deepcopy

#Sort columns by null_value counts
def Sort(sub_li,reverse): 
  
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of  
    # sublist lambda has been used 
    sub_li.sort(key = lambda x: x[1], reverse = reverse) 
    return sub_li 

#Writing python list to csv
def list2csv(complete_list):
    import csv
    csv_columns = ['Sütun Adı', 'Boş değer Sayısı', 'Boş Değer Yüzdesi']
    csv_file = "null_values_report.csv"
    try:
        with open(csv_file, 'w',newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in complete_list:
                writer.writerow(i)
    except IOError:
        print("I/O error")

#Getting null data report of data
def null_info_report(data):
    null_info = []
    for column in data.columns:
        null_count = data[column].isnull().sum(axis = 0)
        null_percentage = (data[column].isnull().sum(axis = 0) / len(data[column])) * 100
        null_percentage2 = (data[column].isnull().mean()) * 100
        null_info.append([column,null_count, null_percentage, null_percentage2])
    null_info = Sort(null_info, reverse=True)
    return null_info

#Write csv from pandas dataframe with encoding
def write_encoded(data):
    pp.encode()
    data = pp.getData()
#    data.to_csv('variants_encoded.csv', index=False)
    data.to_csv('variants_encoded_only_VUS.csv', index=False)
    return data
    
#Write csv from pandas dataframe without encoding
def write_not_encoded(data):
    # data.to_csv('variants_not_encoded_only_VUS.csv', index=False)
    data.to_csv('variants_not_encoded.csv', index=False)


#Data Insertion
pp = preprocessing.Preprocess(data_path_c='dict2csv/variants.csv',read_dtype={'motif.ehipos':object, 'motif.ename':object, 'cadd.istv':object},autoEliminateNullColumns=False,autoImpute=False) #https://stackoverflow.com/a/27232309/8149411
data = pp.getData()
data = data.astype({'motif.ehipos':np.bool, 'motif.ename':np.bool, 'cadd.istv':str})

#Drop id and url columns
pp.dropCols(['_id','cadd._license','clinvar._license','clinvar.rsid','_score'])
data = pp.getData()

#Drop id and url columns for only_vus datas
# pp.dropCols(['cadd._license','clinvar._license','clinvar.rsid','_score']) 
# data = pp.getData()

all_data_null_count = data.isnull().sum().sum()
print('\nall_data_null_count',str(all_data_null_count))
null_report = null_info_report(data)
print('\nnull_report:',null_report)
list2csv(null_report)

pp.eliminateNullColumns()
data = pp.getData()


pp.impute()
data = pp.getData()

### fix the data in incorrect format 
data['cadd.isderived'] = data['cadd.isderived'] == 1
data['cadd.istv'] = data['cadd.istv'] == 'TRUE'

### Dropping rows in correct format at "rcv.clinical_significance" column for training-testing datas
data = data[data['rcv.clinical_significance'] != 'Uncertain significance']
data = data[data['rcv.clinical_significance'] != 'not provided']
data = data[data['rcv.clinical_significance'] != 'other']
data = data[data['rcv.clinical_significance'] != 'risk factor']
data = data[data['rcv.clinical_significance'] != 'Conflicting interpretations of pathogenicity']
data = data[data['rcv.clinical_significance'] != 'Benign/Likely benign']
data = data[data['rcv.clinical_significance'] != 'Pathogenic/Likely pathogenic']


### Dropping rows in correct format at "rcv.clinical_significance" column for prediction datas (only_vus datas)
# data = data[data['rcv.clinical_significance'] == 'Uncertain significance'] #only VUS

#Reset index of dataframe before writing csv
data = data.reset_index(drop=True)
pp.setData(data)
data = pp.getData()


write_not_encoded(data)
data = write_encoded(data)
print(data.info())


