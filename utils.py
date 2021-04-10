import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

def SetDateTimeIndex(df, label):
    df = df.set_index(label)
    df.index = pd.to_datetime(df.index)
    return df


def TopBrands(df_ds, minJobs):
    # I want to investigate the monthly absolute number of job openings to get  an idea of the major players / recruiters
    ds_brand_count_permonth = pd.DataFrame({'count' : df_ds.sort_index().groupby(['year','month',"brand"]).brand.count()}).reset_index()
    ds_brand_count_permonth['year'] = ds_brand_count_permonth['year'].apply(str)
    ds_brand_count_permonth['month'] = ds_brand_count_permonth['month'].apply(str)
    ds_brand_count_permonth['date'] = ds_brand_count_permonth['year'] + '-' + ds_brand_count_permonth['month']
    ds_brand_count_permonth = ds_brand_count_permonth.set_index('date')
    ds_brand_count_permonth.index = pd.to_datetime(ds_brand_count_permonth.index)
    topBrands_month = ds_brand_count_permonth[ds_brand_count_permonth['count'] >= minJobs]
    
    ds_brand_count = pd.DataFrame({'count' : df_ds.sort_index().groupby([df_ds.index,"brand"]).brand.count()}).reset_index()
    ds_brand_count = SetDateTimeIndex(ds_brand_count,'as_of_date')
    
    topBrands_day = ds_brand_count[ds_brand_count['brand'].isin(np.unique(topBrands_month['brand']).tolist())]
    
    return topBrands_month,topBrands_day


def SelectDataJobs(df):
	df_ds = df.loc[df.title.str.contains('Data Scientist', na=False),:]
	df_ds = SetDateTimeIndex(df_ds,'as_of_date')
	df_ds['month'] = df_ds.index.month
	df_ds['year'] = df_ds.index.year
	return df_ds


def plotTopBrands(topBrands_day):
	a4_dims = (11.7, 8.27)
	fig, ax = plt.subplots(figsize=a4_dims)
	sns.lineplot(x=topBrands_day.index, y= topBrands_day['count'], hue = 'brand', data=topBrands_day)
	ax.set_title('Job postings per day for top Data Science seeking companies, January 2017 - July 2018')
	ax.set_ylabel('Number of Job Postings')
	ax.set_ylabel('Date')

def predictJobs(df,lag = 0):
	print('model with lag: ', lag)
	merge_Apple = df
	print(merge_Apple.head())
	merge_Apple['Open_lag'] = merge_Apple['Open'].shift(lag)
	print(merge_Apple.head())
	merge_Apple = merge_Apple.dropna()

	train_data, test_data = merge_Apple[0:int(len(merge_Apple)*0.7)], merge_Apple[int(len(merge_Apple)*0.7):]
	
	X_train = train_data['Open_lag'].values.reshape(-1, 1)
	X_test = test_data['Open_lag'].values.reshape(-1, 1)
	y_train = train_data['count'].values.reshape(-1, 1)
	Y_test = test_data['count'].values.reshape(-1, 1)

	model = sm.OLS(y_train,X_train, missing='drop')
	result = model.fit()
	ypred = result.predict(X_test)
	
	print(result.summary())
	
	c, p = stats.pearsonr(ypred.flatten(), Y_test.flatten())
	print(f"Correlation between Predicted and Observed Jobs: {c}\n")
	plotPrediction(ypred,Y_test)
	
	return result, ypred


def plotPrediction(ypred,Y_test):
	a4_dims = (11.7, 8.27)
	fig, ax = plt.subplots(figsize=a4_dims)
	sns.scatterplot(x = ypred.flatten(), y = Y_test.flatten())
	ax.set_title('Correlation between prediction by stock opening stock price and job postings')
	ax.set_xlabel('Predicted Jobs')
	ax.set_ylabel('Observed Jobs')
