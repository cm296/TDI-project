import time
import pandas as pd
import numpy as np
import utils as ut

def loadCSV(chunksize=1_000_000,limitChunkIter=[]):
    # #To load a large dataset, I'm using panda's chunk function and selecting only the columns I'm interested in, while also eliminating rows that have missing brand information 
#     chunksize = 1_000_000
#     limitChunkIter = []
    start = time.time()
    df = pd.DataFrame()
    filenumber = 0
    #read data in chunks of 1 million rows at a time
    for parti in range(1,8):
        
        filenumber = filenumber+1
        print('loading file number', filenumber)
        for i, chunk in enumerate(pd.read_csv('../datasets/temp_datalab_records_job_listings_'+str(parti)+'.csv',  usecols = ['as_of_date','title','category','locality','region','country','brand'], parse_dates=['as_of_date'], dtype={'category':'category','title':'category','locality': 'category','region':'category','country':'category'} ,chunksize=chunksize)):
            
#             print('file number', filenumber,'chunk ', i)
            #skipping if all brand info is empty for this chunk
            if chunk[~chunk['brand'].isna()].empty:
                print('skipping')
                continue
            else:
                #Cleaning Up dataset
                #ELiminating brand names that are only numbers
                chunk.dropna(subset=['brand'],axis=0, inplace=True)
                chunk = chunk [chunk['country'] == 'USA']
                
                chunk = chunk[~chunk.brand.str.isdigit()]
                df = df.append(chunk)
            #go to next dataset file or terminate if reached the chunk iteration limit
            if limitChunkIter:
                if i ==limitChunkIter:
                    break
    end = time.time()
    del chunk
    print("Read csv with chunks: ",(end-start),"sec")
    df.to_csv('../temp_datalab_records_job_listings_nona.csv')
    return df


def MergeStocks(df):
    stocks = pd.read_csv('AAPL.csv')
    stocks = ut.SetDateTimeIndex(stocks, 'Date')
    merge = pd.merge(df,stocks, how='inner', left_index=True, right_index=True)
    return merge