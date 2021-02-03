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
#%%
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
#Parallelize data to 24 sub-series
raw_data = true_load.loc["2019","SICI"] #test data
parallel_data = parallelize(raw_data)
#%%
data = pd.Series(parallel_data[12], index = pd.date_range(start = "2019-01-01 12:00", end = "2019-12-31 12:00", tz = "UTC", freq = "D")) #test midday load
train, validate, test = data[:"2019-09"], data["2019-10"], data["2019-11"]
#%%
#Compute predictors for values in validate
X = [] #Empty list to store predictors
for ind in validate.index:
	predictor = raw_data[ind - pd.Timedelta(168, "H"):ind - pd.Timedelta(1, "H")] #previous week
	X.append(predictor)		
			    
X = np.array(X) #sklearn format
y_validate = validate.values
#%%
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR
from sklearn.feature_selection import SequentialFeatureSelector
model = LinearRegression()
sfs = SequentialFeatureSelector(model, 
								n_features_to_select=20,
								direction = 'forward',
								n_jobs=-1)
sfs.fit(X,y_validate) #fit sequential feature selector
selected_features = sfs.get_support(indices = True) #get selected features
selected_mask = sfs.get_support()

#%%
#Compute train and test set
X_test = []
for ind in test.index:
	predictor = raw_data[ind - pd.Timedelta(168, "H"):ind - pd.Timedelta(1, "H")] #previous week
	X_test.append(predictor)
X_test = np.array(X_test)
y_test = test.values
X_train = np.array([raw_data[ind - pd.Timedelta(168, "H"):ind - pd.Timedelta(1, "H")] for ind in train.index[7:]])
y_train = train.values[7:]
#%%
#Complete
complete_model = LinearRegression().fit(X_train, y_train)
y_hat_complete = complete_model.predict(X_test)
#Restricted Model
restrict_model = LinearRegression().fit(X_train[:,selected_mask], y_train)
y_hat_restricted = restrict_model.predict(X_test[:,selected_mask])
#%%
#Plot results
plt.plot(y_test, label = 'real', linewidth=2, color='black', alpha = 0.75)
plt.plot(y_hat_complete, linewidth=1, color='red', linestyle='--', label = 'all features')
plt.plot(y_hat_restricted, linewidth=1, color = 'blue', linestyle='-.', label = 'restricted features')
plt.title('Sequential feature selector test')
plt.legend()
plt.show()
#%%
print("Complete model error: ", mape(y_test, y_hat_complete))
print("Restricted Model error: ", mape(y_test, y_hat_restricted))

#%%
#Test Temperature Feature Selection
temp = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/temperatures.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True
					) 
temp = temp.tz_convert("UTC")
raw_temp = temp.loc["2019", "NORD"]
parallel_temp = parallelize(raw_temp)
#%%
single_hour_temp = pd.Series(parallel_temp[12], index = pd.date_range(start = "2019-01-01 12:00", end = "2019-12-31 12:00", tz = "UTC", freq = "D"))
temp_train, temp_validate, temp_test = single_hour_temp[:"2019-09"], single_hour_temp["2019-10"], single_hour_temp["2019-11"]
TX_train = np.array([raw_temp[ind - pd.Timedelta(168, "H"):ind - pd.Timedelta(1, "H")] for ind in temp_train.index[7:]])
TX_validate = np.array([raw_temp[ind - pd.Timedelta(168, "H"):ind - pd.Timedelta(1, "H")] for ind in temp_validate.index])
TX_test = np.array([raw_temp[ind - pd.Timedelta(168, "H"):ind - pd.Timedelta(1, "H")] for ind in temp_test.index])
#%%
model = LinearRegression()
sfs = SequentialFeatureSelector(model, 
								n_features_to_select=20,
								direction = 'forward',
								n_jobs=-1)
sfs.fit(TX_validate, y_validate)
selected_temps = sfs.get_support()

#%%
#Complete
complete_model = LinearRegression().fit(TX_train, y_train)
y_hat_complete = complete_model.predict(TX_test)

#Restricted Model
restrict_model = LinearRegression().fit(TX_train[:,selected_temps], y_train)
y_hat_restricted = restrict_model.predict(TX_test[:,selected_temps])

#Plot results
plt.plot(y_test, label = 'real', linewidth=2, color='black', alpha = 0.75)
plt.plot(y_hat_complete, linewidth=1, color='red', linestyle='--', label = 'all features')
plt.plot(y_hat_restricted, linewidth=1, color = 'blue', linestyle='-.', label = 'restricted features')
plt.title('Sequential feature selector test - temps')
plt.legend()
plt.show()

#Error
print("Complete model error: ", mape(y_test, y_hat_complete))
print("Restricted Model error: ", mape(y_test, y_hat_restricted))

#%%
#Join models: temps + load
JX_train = np.concatenate([X_train, TX_train], axis = 1)
JX_test = np.concatenate([X_test, TX_test], axis = 1)
J_mask = np.concatenate([selected_mask, selected_temps])
#Complete
complete_model = LinearRegression().fit(JX_train, y_train)
y_hat_complete = complete_model.predict(JX_test)

#Restricted Model
restrict_model = LinearRegression().fit(JX_train[:,J_mask], y_train)
y_hat_restricted = restrict_model.predict(JX_test[:,J_mask])

#Plot results
plt.plot(y_test, label = 'real', linewidth=2, color='black', alpha = 0.75)
plt.plot(y_hat_complete, linewidth=1, color='red', linestyle='--', label = 'all features')
plt.plot(y_hat_restricted, linewidth=1, color = 'blue', linestyle='-.', label = 'restricted features')
plt.title('Sequential feature selector test - load + temp')
plt.legend()
plt.show()

#Error
print("Complete model error: ", mape(y_test, y_hat_complete))
print("Restricted Model error: ", mape(y_test, y_hat_restricted))