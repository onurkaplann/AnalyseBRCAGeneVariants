# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:49:34 2020

@author: EmreKARA
"""
import pandas as pd
import json
import copy
from collections import Counter
import csv

def get_all_values(nested_dictionary,counter,preKey=None):
    for key, value in nested_dictionary.items():
        if type(value) is dict:
            get_all_values(value,counter,preKey=key)
        else:
            if type(value) is list:
                freq_list = []
                for i in value:
                    freq_list.append(i['clinical_significance'])
                most_frequent = Counter(freq_list).most_common(1)
                myDict['rcv.clinical_significance'] = most_frequent[0][0]
            else:
                if preKey is not None:
                    myDict[(preKey+'.'+key)] = value
                else:
                    myDict[key] = value

def unique(list1): 
      
    list_set = set(list1) 
    unique_list = (list(list_set)) 
    return unique_list


data = pd.read_json("cadd_clinvar.json") #GetDataFromJson
with open('cadd_clinvar.json', 'r') as openfile: 
        dataJson = json.load(openfile)

dataJsonPre = copy.deepcopy(dataJson)
for i in dataJson: #RemoveMultiGeneRecords
    if type(i['cadd']['gene']) is list:
        genesList = copy.deepcopy(i['cadd']['gene'])
        for j in range(len(genesList)):
            tempDict = dict(genesList[j])
            try:
                if genesList[j]['genename'] == 'BRCA1' or genesList[j]['genename'] == 'BRCA2':
                    try:
                        i['cadd']['annotype'] = i['cadd']['annotype'][j]
                    except:
                        pass
                    try:
                        i['cadd']['consdetail'] = i['cadd']['consdetail'][j]
                    except:
                        pass
                    try:
                        i['cadd']['consequence'] = i['cadd']['consequence'][j]
                    except:
                        pass
                    try:
                        i['cadd']['consscore'] = i['cadd']['consscore'][j]
                    except:
                        pass
                    try:
                        i['cadd']['gene'] = i['cadd']['gene'][j]
                    except:
                        pass
                    try:
                        i['cadd']['exon'] = i['cadd']['exon'][j]
                    except:
                        pass
                    try:
                        i['cadd']['dst2splice'] = i['cadd']['dst2splice'][j]
                    except:
                        pass
                    try:
                        i['cadd']['dst2spltype'] = i['cadd']['dst2spltype'][j]
                    except:
                        pass
                    
            except:
                pass

extracted = [] #Extract Nested Dicts
counter = 0
for i in dataJson:
    myDict = {}
    get_all_values(i,counter)
    extracted.append(copy.deepcopy(myDict))
    counter += 1

allFeatures = [] #GetAllUniqueFeatures
for i in extracted:
    for k in i.keys():
        allFeatures.append(k)
uniqueFeatures = unique(allFeatures)


for i in extracted: #All Unique Features created in all Dicts
    for u in uniqueFeatures:
        try:
            i[u] = i[u]
        except KeyError:
            i[u] = None


uniqueFeatures.sort()  #Write Data To CSV
csv_columns = uniqueFeatures
csv_file = "variants.csv"
try:
    with open(csv_file, 'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i in extracted:
            writer.writerow(i)
except IOError:
    print("I/O error")





