#Experimental Linear Regression Test with local feature selection
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from utils_lib import  T_cr, T_hr, T_xhr, parallelize, unparallelize
import pandas as pd
import numpy as np

class LinRegFSModel:
	"""Model to produce short-term load forecasts, using feature selection"""
	#---------------------------Public Methods------------------------------
	def __init__(self,
				 terna_series,
				 corporate_series,
				 temp_series = None,
				 solar_series = None,
				 holiday_series = None,
				 lockdown_series = None,
				 freq = "H"
				 ):
		"""Initialize forecasts model.
		Parameters.
		terna_series: pd.Series -- Terna bills load Series
		corporate_series: pd.Series -- Corporate forecasts Series 
		freq: str. One of "H", "D". Default: "H" - hourly.
		"""
		#Data - Load
		self._terna_series = terna_series
		self._corporate_series = corporate_series
		#Data - Placeholders
		self._train = None
		self._validate = None
		self._test = None 
		self._load = None #Complete load series (train + validation + test)
		#Data - Exog
		
		self._temp_series = temp_series
		self._cooling_requirement, self._heating_requirement, self._extra_heating_requirement = self._transform_temperatures()
		self._solar_series = solar_series
		self._holiday_series = holiday_series
		self._lockdown_series = lockdown_series
		
		#Parameters 
		self._freq = pd.Timedelta(1, freq) #frequency of TS
		#Parallelize Load: Generate 24 series, one per each hour of day
		self._parallel_train = None
		self._parallel_validate = None
		self._parallel_test = None

	def fit_predict(self, test_time_range, validation_lag = pd.Timedelta(30, 'D')):
		"""Generate & Fit submodels; generate forecast of test_time_range.
		Return pd.Series.
		Wrapper around _sub_fit_predict
		Parameters:		
		test_time_range: daterange. Return full forecast.
		validation_lag: timedelta. Breadth of validation window. Only days.
		"""
		#----------------------------Split Datasets---------------------------
		self._split_data(test_time_range) #Generate train, test & validation sets
		n_days = int(len(test_time_range)/24) #Number of days to forecast
		#---------------------------Forecast routine--------------------------
		forecast_array = np.empty((24, n_days)) #Empty array to store forecasts
		for t in range(24):
			forecast_array[t,:] = self._sub_fit_predict(t) #row t
		forecast = unparallelize(forecast_array, index = test_time_range) #build series
		return forecast

	#-----------------------------Non-Public Methods-------------------------
	def _sub_fit_predict(self, t):
		"""Return Hourly forecast for specified t (0..23) of horizon determined by self._test"""
		#----------------(Load) Validation FS----------------
		#Generate validation set
		print("Load FS model for hour ", t)
		y_validate = self._parallel_validate[t,:]
		validation_index = [self._validate.index[i] for i in range(t, len(self._validate), 24)] #retrieve time indices of target varialbes
		X_validate = [self._compute_load_predictor(ix) for ix in validation_index] #compute predictors
		X_validate = np.stack(X_validate) #Cast as numpy array
		validation_model = LinearRegression() #Instantiate validation model
		sfs = SequentialFeatureSelector(validation_model,
										n_features_to_select = 12,
										direction = 'forward',
										n_jobs = -1)
		sfs.fit(X_validate, y_validate) #Perform feature selection over validation set
		selected_loads = sfs.get_support() #Boolean Mask with selected features
		#------------------------Temperature Feature Selection---------------
		print("Temp FS model for hour ", t)
		sfs = SequentialFeatureSelector(validation_model,
								n_features_to_select = 5,
								direction = 'forward',
								n_jobs = -1)
		X_validate = [self._compute_temp_predictor(ix) for ix in validation_index]
		sfs.fit(X_validate, y_validate) #Feature selection for temperatures
		selected_temps = sfs.get_support()
		#-----------------------------------Train----------------------------
		print("Train & predict")
		final_model = LinearRegression()
		#final_model = SVR(kernel = 'poly', degree = 3, C = 1, gamma=1.0,  epsilon=1, tol = 1)
		y_train = self._parallel_train[t, 3:] #skip first 3 days
		train_index = [self._train.index[i] for i in range(t + 72, len(self._train), 24)] #retrieve training targets'indices
		X_load = [self._compute_load_predictor(ix) for ix in train_index]
		X_load = np.stack(X_load)[:, selected_loads] #restrict to selected loads
		X_temp = [self._compute_temp_predictor(ix) for ix in train_index]
		X_temp = np.stack(X_temp)[:, selected_temps] #restrict to selected temperatures.
		X_train = np.concatenate([X_load, X_temp], axis = 1)
		final_model.fit(X_train, y_train)
		#-----------------------Predict--------------------
		test_index = [self._test.index[i] for i in range(t, len(self._test), 24)]
		X_test = [self._compute_restricted_predictor(ix, selected_loads, selected_temps) for ix in test_index]
		X_test = np.stack(X_test) #restrict to selected features
		forecast = final_model.predict(X_test)
		return forecast

	def _transform_temperatures(self):
		"""Transform temperature series to three new series: cooling requirement,
		heating requirement, extra heating requirement."""
		cooling = self._temp_series.apply(T_cr)
		heating = self._temp_series.apply(T_hr)
		extra_heating = self._temp_series.apply(T_xhr)
		return cooling, heating, extra_heating


	def _split_data(self, test_time_range, validation_lag = pd.Timedelta(30, 'D')):
		"""Split load dataset into train set, validation set and test set,
		for specified test time range. Generate also parallelized representations.
		Parameters:
		test_time_range: pd.daterange.
		validation_lag: pd.Timedelta. Validation set size. default: 1 month
		"""
		test_0 = test_time_range[0] #First test time
		test_1 = test_time_range[-1] #Last test time
		self._test = self._corporate_series[test_0:test_1] #test set
		self._parallel_test = parallelize(self._test) #paralllelized representation
		validate_0 = test_0 - validation_lag #First validation time
		validate_1 = test_0 - self._freq #Last validation time
		self._validate = self._corporate_series[validate_0:validate_1] #validation set
		self._parallel_validate = parallelize(self._validate) 
		train_1 = validate_0 - self._freq #Last train time
		self._train = self._terna_series[:train_1] #train set
		self._parallel_train = parallelize(self._train)
		self._load = pd.concat([self._train, self._validate, self._test]) #Complete load series
		self._parallel_load = parallelize(self._load)

	def _compute_load_predictor(self, target_index, look_back = pd.Timedelta(72, 'H')):
		"""Compute load predictor variables for given response variable
		with index target_index"""
		predictor = self._load[target_index - look_back: target_index - pd.Timedelta(1, "H")] #retrieve load values
		return predictor.values

	def _compute_temp_predictor(self, target_index, look_back = pd.Timedelta(24,'H')):
		"""Compute temperature-related predictor variables for given response variable
		with index target_index"""
		start = target_index - look_back
		end = target_index + pd.Timedelta(23, "H")
		cooling = self._cooling_requirement[start:end]
		heating = self._heating_requirement[start:end]
		extra_heating = self._extra_heating_requirement[start:end]
		predictor = np.concatenate([cooling, heating, extra_heating])
		return predictor

	def _compute_restricted_predictor(self, target_index, selected_loads, selected_temps):
		"""Compute predictor with restriction masks"""
		temp = self._compute_temp_predictor(target_index)[selected_temps]
		load = self._compute_load_predictor(target_index)[selected_loads]
		predictor = np.concatenate([load, temp])
		return predictor



