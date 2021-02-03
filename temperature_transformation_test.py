#%%Test Temperature Feature Selection
import pandas as pd
import numpy as np
import sys
sys.path.append("D:/Users/F.Moraglio/Documents/python_forecasting/stage_5/")
from utils_lib import *
#Plots
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
plt.ioff()
register_matplotlib_converters()
sns.set_style('darkgrid')
plt.rc('figure',figsize=(16,12))
plt.rc('font', size=16)
#%%Datasets
temp = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/temperatures.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True
					) 
temp = temp.tz_convert("UTC")

true_load = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/fatture.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True
					)
true_load = true_load.tz_localize("Europe/Rome", ambiguous="infer")
true_load = true_load.tz_convert("UTC") 
#%%
#Params 
hour = 12
#%%
#Generate temperature-based feature series
sample_temp = temp.loc["2019", "NORD"]
tcr = pd.Series(np.vectorize(T_cr)(sample_temp), index = sample_temp.index, tz="UTC") #cooling requirement
thr = pd.Series(np.vectorize(T_hr)(sample_temp), index = sample_temp.index, tz="UTC") #heating requirement
txhr = pd.Series(np.vectorize(T_xhr)(sample_temp), index = sample_temp.index, tz="UTC") #extra heating requirement
#%%
#Load data
sample_load = true_load.loc["2019", "NORD"]
single_hour_load = pd.Series(parallelize(sample_load)[hour],index = pd.date_range(start = "2019-01-01 12:00", end = "2019-12-31 12:00", tz = "UTC", freq = "D"))
train, validate, test = single_hour_load[:"2019-09"], single_hour_load["2019-10"], single_hour_load["2019-11"]
y_train, y_validate, y_test = train.values[3:], validate.values, test.values
#%%
#Generate temperature predictors: 3 days before plus one day ahead -> 96 variables for each indicator (3x96 = 288)
XT_validate = []
for ind in validate.index:
	date = ind.date()
	p_cr = tcr[date - pd.Timedelta(3,'D'):date + pd.Timedelta(1, 'D')]
	p_hr = thr[date - pd.Timedelta(3,'D'):date + pd.Timedelta(1, 'D')]
	p_xhr = txhr[date - pd.Timedelta(3,'D'):date + pd.Timedelta(1, 'D')]
	predictor = np.concatenate([p_cr, p_hr, p_xhr])
	XT_validate.append(predictor)
XT_validate = np.array(XT_validate)

XT_train = []
for ind in train.index[3:]:
	date = ind.date()
	p_cr = tcr[date - pd.Timedelta(3,'D'):date + pd.Timedelta(1, 'D')]
	p_hr = thr[date - pd.Timedelta(3,'D'):date + pd.Timedelta(1, 'D')]
	p_xhr = txhr[date - pd.Timedelta(3,'D'):date + pd.Timedelta(1, 'D')]
	predictor = np.concatenate([p_cr, p_hr, p_xhr])
	XT_train.append(predictor)
XT_train = np.array(XT_train)

XT_test = []
for ind in test.index:
	date = ind.date()
	p_cr = tcr[date - pd.Timedelta(3,'D'):date + pd.Timedelta(1, 'D')]
	p_hr = thr[date - pd.Timedelta(3,'D'):date + pd.Timedelta(1, 'D')]
	p_xhr = txhr[date - pd.Timedelta(3,'D'):date + pd.Timedelta(1, 'D')]
	predictor = np.concatenate([p_cr, p_hr, p_xhr])
	XT_test.append(predictor)
XT_test = np.array(XT_test)
#%%
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression 
model = LinearRegression()
sfs = SequentialFeatureSelector(model, 
								n_features_to_select=20,
								direction = 'forward',
								n_jobs=-1)
sfs.fit(XT_validate, y_validate) #fit sequential feature selector
selected_mask = sfs.get_support()
#%%
#Complete
complete_model = LinearRegression().fit(XT_train, y_train)
y_hat_complete = complete_model.predict(XT_test)
#Restricted Model
restrict_model = LinearRegression().fit(XT_train[:,selected_mask], y_train)
y_hat_restricted = restrict_model.predict(XT_test[:,selected_mask])
#%%
#Plot results
plt.plot(y_test, label = 'real', linewidth=2, color='black', alpha = 0.75)
plt.plot(y_hat_complete, linewidth=1, color='red', linestyle='--', label = 'all features')
plt.plot(y_hat_restricted, linewidth=1, color = 'blue', linestyle='-.', label = 'restricted features')
plt.title('Sequential feature selector test')
plt.legend()
plt.show()