'''
Machine learning API
'''
import numpy as np
import tensorflow as tf
import random

import os
##### For Reproducibility #####
os.environ['PYTHONHASHSEED']= '0'

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)

import keras
from keras import backend as K
session_conf= tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess= tf.Session(graph= tf.get_default_graph(), config= session_conf)
K.set_session(sess)

###############################

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import pandas as pd

import sys

import matplotlib.pyplot as plt


class ML(object):

	# Dunders
	def __init__(self, df= None, name= "ML", labels= None, validation_size= None, test_size= None, thresh= None):
		self.name= name
		self.df= df
		self.labels= labels

		self.validation_size= validation_size
		self.test_size= test_size
		self.thresh= thresh

		self.true_max, self.true_min, self.true_mean, self.true_sd= self.df.max(), self.df.min(), self.df.mean(), self.df.std()

		self.was_standardized, self.was_normalized= False, False


	
	def __repr__(self): # Gives some information on the dataframe
		return("DataFrame with columns {} and shape {} to be trained for predicting labels {}, using {}.".format(self.df.columns, self.df.shape, self.labels, self.name))


	def __len__(self): # Gives the number of rows
		return self.df.shape[0]


	# Getters
	@property
	def df(self):
		return self._df
	
	@property
	def labels(self):
		return self._labels
	
	@property
	def test_size(self):
		return self._test_size

	@property
	def validation_size(self):
		return self._validation_size


	# Setters
	@df.setter
	def df(self, n): #"n" here is a df
		self._df= n

	@labels.setter
	def labels(self, n): #"n" here is a list of those columns in the df you want to use as labels
		self._labels= n

	@test_size.setter
	def test_size(self, n): #"n" here is a float between 0 and 1 with two values after the decimal
		self._test_size= n

	@validation_size.setter
	def validation_size(self, n): #"n" here is a float between 0 and 1 with two values after the decimal
		self._validation_size= n


	# API methods
	def scatter(self):
		cols= self.df.drop(self.labels, axis=1).columns
		for i in cols:
			for j in self.labels:
				plt.scatter(self.df[i], self.df[j])
				plt.title("{} vs {}".format(i, j))
				plt.xlabel("{}".format(i))
				plt.ylabel("{}".format(j))
				plt.show()

				plt.clf()

	def denormalize_col(self, cols= []):
		for col in cols:
			self.df[col]= (self.df[col]*(max_dict[col]-min_dict[col]))+min_dict[col]

	def rescale_col(self, cols= []):
		for col in cols:
			self.df[col]= self.df[col]*self.mul*self.std_dict[col]+self.mean_dict[col]


	def scale_df(self, cols= [], mul= 1):
		self.was_standardized= True
		self.mul= mul
		self.mean_dict= {}
		self.std_dict= {}

		if not cols:
			for col in self.df.columns:
				self.mean_dict[col]= self.df[col].mean()
				self.std_dict[col]= self.df[col].std()

				self.df[col]= ((self.df[col] - self.df[col].mean())/self.df[col].std())/self.mul

		else:
			for cols in self.df.columns:
				self.mean_dict[col]= self.df[col].mean()
				self.std_dict[col]= self.df[col].std()

				self.df[col]= ((self.df[col] - self.df[col].mean())/self.df[col].std())/self.mul


	def normalize_df(self, cols= []): # Normalizes DF between 0 and 1
		self.was_normalized= True
		self.min_dict= {}
		self.max_dict= {}

		if not cols:
			for col in self.df.columns:
				self.min_dict[col]= self.df[col].min()
				self.max_dict[col]= self.df[col].max()

				self.df[col]= ((self.df[col]-self.df[col].min())/(self.df[col].max()-self.df[col].min()))

		else:
			for col in cols:
				self.min_dict[col]= self.df[col].min()
				self.max_dict[col]= self.df[col].max()

				self.df[col]= ((self.df[col]-self.df[col].min())/(self.df[col].max()-self.df[col].min()))


	def standardize_df(self):
		self.df= preprocessing.scale(self.df)
		self.was_standardized= True


	def parameters(self): #Pass a string with the format: 
		print("Validation size: {}\nTest size: {}\nLabels: {}".format(self.validation_size, self.test_size, self.labels))


	def to_categorical(self, cols= []):
		for col in cols:
			self.df[col]= pd.Categorical(self.df[col])


	def one_hot_encode(self, k= 'k-1'):
		if k== 'k-1':
			self.df= pd.get_dummies(self.df, prefix= 'one_hot', drop_first= True)
		else:
			self.df= pd.get_dummies(self.df, prefix= 'one_hot')


	def val_dataset(self):
		# X_train, X_val, y_train, y_val
		self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size= self.validation_size, shuffle= True, random_state= 42)


	def train_test_dataset(self):
		# df with labels: df_Y, df with dataset to train on: df_X
		self.df_y= self.df[self.labels]
		self.df_X= self.df.drop(self.labels, axis= 1)

		# X_test, y_test
		self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(self.df_X.values, self.df_y.values, test_size=self.test_size, shuffle= True, random_state= 42)

		# Val dataset
		self.val_dataset()


	# def int_tup_from_tuple(self, tup= None):
	# 	for i 
