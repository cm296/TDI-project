# DI-project
#### The goal of this project was to test whether fluctuations in a company's stock price could be predictive of that company's job postings in the field of Data Science. Identifying such a relationship would be helpful for job seekers and employment agencies.

## Part1_StocksJobs_Apple.ipynb: 
#### - The notebook loads job posting data for all NYSE and Nasdaq stock , cleans them, remove missing data, and saves them in a new csv file. 

#### - It then identifies the major companies that posted job openings in the field of "Data Science" and, filters data to only show companies that posted a high number of jobs (>=500 for at least one month during the time period). The company for which data seemed more complete, Apple, was selected for further analysis.

#### - It then loads a dataset of the stock prices and merges it with the job posting based on time points

#### - finally, it runs a predictive model with X as (lagged) stock price and Y as number of Job postings, using a simple ordinary least square regression. 

#### - Result shows decent prediction of job openings in data science of stock prices with 7-day lag, but further observation with different companies, models and lags are needed. 


# Setting up directories:
#### To load the dataset, the notebook uses the python module loadCSV.py , which assumes a directory with path '../datasets' where all the .csv files for the job openings are located. filenames are saved, with filename temp_datalab_records_job_listings_1.csv,*_2.csv,*_3.csv, and so on until 7


# Environment: 
### Python 3
#### Packages: pandas, numpy, matplolib, seaborn, statsmodels, scipy

