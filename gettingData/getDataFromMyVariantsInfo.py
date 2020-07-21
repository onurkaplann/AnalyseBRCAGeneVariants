# -*- coding: utf-8 -*-
"""
Created on Sun March  15 06:12:22 2020

@author: EmreKARA
"""
import json
import pymongo
import myDB
import requests 

#åDownloading Data from myvariant.info by web API
def downloadData():
    URL = "http://myvariant.info/v1/query"
    #params = 'q=dbsnp.gene.symbol:BRCA*&fields=dbsnp.gene.symbol&fetch_all=TRUE'#
#    params = 'q=dbnsfp.genename:BRCA* OR dbsnp.gene.symbol:BRCA* OR clinvar.gene.symbol:BRCA* OR evs.gene.symbol:BRCA* OR cadd.gene.genename:BRCA* OR wellderly.gene:BRCA*&fetch_all=TRUE'#
    params = 'q= cadd.gene.genename:BRCA* AND _exists_:clinvar &fields=cadd,clinvar.rsid,clinvar.rcv.clinical_significance&fetch_all=TRUE'
    data = []

    counter = 0
    while True:
        try:
            r = requests.get(url = URL, params = params) 
            data_temp = r.json()
            params = 'scroll_id=' + data_temp['_scroll_id']
            data += data_temp['hits']
            print('data loading...', counter)
            counter += 1
        except Exception as err:
            if str(err) == '\'_scroll_id\'':
                print('All the datas are loaded.')
                break;
            else:
                print('err: ', err)
                break;
    data = data[:-1]
    return data

#Write json from pandas dataframe
def writeJson(data, path='databases/variantsdata.json'):
    with open(path, "w") as outfile:
        text = json.dumps(data)
        outfile.write(text)
#Read json file from local
def readJson():
    with open('databases/variantsdata.json', 'r') as openfile: 
        data = json.load(openfile)
    return data

#Write pandas dataframe into MongoDB database
def writeMongoDB(data,database='myvariantsinfo', collection='variants'):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/") 
    mydb = myclient[database] #db
    mycol = mydb[collection] #collection
    y = mycol.insert_many(data)
    return y

#Write pandas dataframe into SQLite database
def write2Sqlite(data):

    db = myDB.sqlite()
    for d in data:
        hits = d['hits']
        variants = []
        for i in hits:
            try:
                gene_symbol = ''
                gene_id = str(i['_id'])
                gene_symbol_pre = i['dbsnp']['gene']
                if len(gene_symbol_pre) > 1:
                    temp = ''
                    for j in gene_symbol_pre:
                        temp += (j['symbol'] + ',')
                    gene_symbol = temp[:-1]
                else:
                    gene_symbol = gene_symbol_pre['symbol']
                variants.append([gene_id, gene_symbol])
            except TypeError as err:
                print('err: ',err)
        db.insert(data = variants)



data = downloadData()
writeMongoDB(data,collection='cadd_clinvar')
writeJson(data, path='databases/cadd_clinvar.json')
#json_data = readJson()



