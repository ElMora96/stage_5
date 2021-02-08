#Testing random forest model
import sys
sys.path.append("D:/Users/F.Moraglio/Documents/python_forecasting/stage_5/")
import lr_fs_lib  as lr #single seasonality prediction
from utils_lib import  mape
import pandas as pd
import numpy as np

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
#Forecast Aziendale
egea_forecast =  pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/forecast.csv",
					sep = ";", #specify separator
					parse_dates = True,
					#NO DAYFIRST!
					decimal=",",
					index_col = 0,
					squeeze = True
					)
egea_forecast.index = pd.to_datetime(egea_forecast.index, utc=True)
egea_forecast = egea_forecast.tz_convert("UTC") 

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

temp = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/temperatures.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True
					) 
temp = temp.tz_convert("UTC")

#Shorten
temp = temp.loc["2018":, "NORD"]
egea_forecast = egea_forecast.loc["2020", "NORD"]
true_load = true_load.loc["2018":, "NORD"]
#%%
model = lr.LinRegFSModel(true_load, egea_forecast, temp)
#%%
test_time_range = pd.date_range(start = '2020-04-01 00:00:00+00:00',
						   end = '2020-04-30 23:00:00+00:00',
						   freq = 'H',
						   tz = 'UTC'
						   )
#%%
true_series = true_load[test_time_range]
egea_series = egea_forecast[test_time_range]
#%%
pred_series = model.fit_predict(test_time_range)
#%%
#Evaluation & Plot Routine
full_err = mape(true_series, pred_series)
egea_err = mape(true_series, egea_series)


plt.plot(true_series, label = "Consuntivo", color = "black", linewidth = 4, alpha = 0.5)
plt.plot(pred_series, label = "Modello", color = "red", linestyle="--", linewidth = 2)
plt.plot(egea_series, label = "Egea", color = "blue", linestyle = "-.", linewidth = 2)
plt.ylabel("Quantità [MWh]")
plt.title("Test Modello LinFS - \nErrore Medio Modello: " + 
		  str(np.round(full_err, 1))+ "%\n Errore Medio Egea (Teorico): " + str(np.round(egea_err, 1)) +"%")
plt.legend()
plt.show()