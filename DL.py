'''
DNN API
Inherits ML
'''

from ML import *


from keras import models
from keras import layers
from keras.layers import Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras import initializers, regularizers
from keras import metrics

from sklearn.metrics import r2_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from scipy import interp

from itertools import cycle

class DL(ML):

	def __init__(self, df= None, name= "DL", labels= None, validation_size= None, test_size= None, n_layers= None, n_nodes_per_layer= None, actvn_per_layer= None, lr_rate= None, optimizer= None, loss= None, metrics= None, epochs= None, batch_size= None, thresh= None, dropout= None, l1= None, l2= None): # df: dataframe to train on, label: label column/s in the dataset-> this is a list of column names
		super().__init__(df= df, name= name, labels= labels, validation_size= validation_size, test_size= test_size, thresh= thresh)
		self.n_layers= n_layers
		self.n_nodes_per_layer= n_nodes_per_layer
		self.actvn_per_layer= actvn_per_layer

		self.lr_rate= lr_rate
		self.optimizer= optimizer
		self.loss= loss
		self.metrics= metrics
		self.epochs= epochs
		self.batch_size= batch_size

		self.dropout= dropout
		self.l1= l1
		self.l2= l2


	# Getters
	@property
	def n_layers(self):
		return self._n_layers
	
	@property
	def n_nodes_per_layer(self):
		return self._n_nodes_per_layer

	@property
	def actvn_per_layer(self):
		return self._actvn_per_layer
	
	@property
	def lr_rate(self):
		return self._lr_rate

	@property
	def optimizer(self):
		return self._optimizer

	@property
	def loss(self):
		return self._loss
	
	@property
	def metrics(self):
		return self._metrics
	
	@property
	def epochs(self):
		return self._epochs

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def dropout(self):
		return self._dropout

	@property
	def l1(self):
		return self._l1

	@property
	def l2(self):
		return self._l2
	


	# Setters
	@n_layers.setter
	def n_layers(self, n): #"n" here is a list of integers
		self._n_layers= n

	@n_nodes_per_layer.setter
	def n_nodes_per_layer(self, n): #"n" here is a list of integers
		self._n_nodes_per_layer= n

	@actvn_per_layer.setter
	def actvn_per_layer(self, n): #"n" here is a list of string
		self._actvn_per_layer= n

	@lr_rate.setter
	def lr_rate(self, n): #"n" here is a scalar
		self._lr_rate= n

	@optimizer.setter
	def optimizer(self, n): #"n" here is a string
		self._optimizer= n

	@loss.setter
	def loss(self, n): #"n" here is a string
		self._loss= n

	@metrics.setter
	def metrics(self, n): #"n" here is a list of string
		self._metrics= n

	@epochs.setter
	def epochs(self, n): #"n" here is an integer
		self._epochs= n

	@batch_size.setter
	def batch_size(self, n): #"n" here is an integer
		self._batch_size= n

	@dropout.setter
	def dropout(self, n): #"n" here is a list
		self._dropout= n

	@l1.setter
	def l1(self, n): #"n" here is an integer
		self._l1= n

	@l2.setter
	def l2(self, n): #"n" here is an integer
		self._l2= n



	# API methods
	def parameters(self): #Pass a string with the format:
		print("Validation size: {}\nTest size: {}\nLabels: {}\nNo. of layers: {}\nNo. of nodes per layer: {}\nActivation function per layer: {}\nLearning rate: {}\nOptimizer: {}\nLoss: {}\nMetrics: {}\nEpochs: {}\nBatch size: {}\nDropout: {}\nl1 regularization: {}\nl2 regularization: {}\n".format(self.validation_size, self.test_size, self.labels, self.n_layers, self.n_nodes_per_layer, self.actvn_per_layer, self.lr_rate, self.optimizer, self.loss, self.metrics, self.epochs, self.batch_size, self.dropout, self.l1, self.l2))


	def set_optimizer(self):

		# Setting optimizer based on the self.optimizer input

		#1. Stochastic Gradient Descent
		if self.optimizer == 'SGD':
			sgd= optimizers.SGD(lr= self.lr_rate)
			return sgd

		#2. RMS prop
		elif self.optimizer == 'RMSprop':
			rmsprop= optimizers.RMSprop(lr= self.lr_rate)
			return rmsprop

		#3. AdaGrad
		elif self.optimizer == 'Adagrad':
			adagrad= optimizers.Adagrad(lr= self.lr_rate)
			return adagrad

		#4. Adadelta
		elif self.optimizer == 'Adadelta':
			adadelta= optimizers.Adadelta(lr= self.lr_rate)
			return adadelta

		#5. Adam
		elif self.optimizer == 'Adam':
			adam= optimizers.Adam(lr= self.lr_rate)
			return adam

		#6. Adamax
		elif self.optimizer == 'Adamax':
			adamax= optimizers.Adamax(lr= self.lr_rate)
			return adamax

		#7. Nadam
		elif self.optimizer == 'Nadam':
			nadam= optimizers.Nadam(lr= self.lr_rate)
			return nadam

		#8. Default
		else:
			print("Adam optimizer with a lr_rate of 0.01 was chosen since no optimizer was provided.")
			return optimizers.Adam(lr= 0.01)


	def plot_loss(self):
		self.loss_values= self.history_dict['loss']
		self.val_loss_values= self.history_dict['val_loss']

		epochs= range(1, len(self.history_dict['loss']) + 1)

		plt.plot(epochs, self.loss_values, 'b', color= 'blue', label= 'Training loss')
		plt.plot(epochs, self.val_loss_values, 'b', color= 'red', label= 'Validation loss')
		plt.title('Training and validation loss')
		# plt.ylim((0, 25))
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		plt.show()

		plt.clf()


	def plot_acc(self):
		self.acc_values= self.history_dict['acc']
		self.val_acc_values= self.history_dict['val_acc']

		epochs= range(1, len(self.history_dict['acc']) + 1)

		plt.plot(epochs, self.acc_values, 'b', label= 'Training acc')
		plt.plot(epochs, self.val_acc_values, 'b', color= 'red', label= 'Validation acc')
		plt.title('Training and Validation accuracy (Classification)')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		plt.show()

		plt.clf()


	def plot_mae(self):
		mae_values= self.history_dict['mean_absolute_error']
		val_mae_values= self.history_dict['val_mean_absolute_error']

		epochs= range(1, len(self.history_dict['mean_absolute_error']) + 1)

		plt.plot(epochs, mae_values, 'bo', label= 'Training mae')
		plt.plot(epochs, val_mae_values, 'b', color= 'red', label= 'Validation mae')
		plt.title('Training and Validation MAE (Regression)')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		plt.show()

		plt.clf()


	def plot_mse(self):
		mse_values= self.history_dict['mean_squared_error']
		val_mse_values= self.history_dict['val_mean_squared_error']

		epochs= range(1, len(self.history_dict['mean_squared_error']) + 1)

		plt.plot(epochs, mse_values, 'bo', label= 'Training mse')
		plt.plot(epochs, val_mse_values, 'b', color= 'red', label= 'Validation mse')
		plt.title('Training and Validation MSE (Regression)')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		plt.show()

		plt.clf()


	def roc_crv(self, macro_roc= False, micro_roc= False):
		n_classes= self.n_nodes_per_layer[-1]
		lw= 2

		# Computing ROC curve and ROC area for each class
		fpr= dict()
		tpr= dict()
		roc_auc= dict()

		for i in range(n_classes):
			fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i].astype(float), self.y_score[:, i])
			roc_auc[i]= auc(fpr[i], tpr[i])

		# Computing micro-average ROC curve and ROC area
		fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.astype(float).ravel(), self.y_score.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


		# Compute macro-average ROC curve and ROC area
		# First aggregate all false positive rates
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(n_classes):
		    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

		# Finally average it and compute AUC
		mean_tpr /= n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

		# Plot all ROC curves
		plt.figure()
		
		if micro_roc:
			plt.plot(fpr["micro"], tpr["micro"],
			         label='micro-average ROC curve (area = {0:0.2f})'
			               ''.format(roc_auc["micro"]),
			         color='deeppink', linestyle=':', linewidth=4)

		if micro_roc:
			plt.plot(fpr["macro"], tpr["macro"],
			         label='macro-average ROC curve (area = {0:0.2f})'
			               ''.format(roc_auc["macro"]),
			         color='navy', linestyle=':', linewidth=4)

		colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
		for i, color in zip(range(n_classes), colors):
		    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
		             label='ROC curve of class {0} (area = {1:0.2f})'
		             ''.format(i, roc_auc[i]))

		plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic for multi-class')
		plt.legend(loc="lower right")
		plt.show()

		plt.clf()



	def classfctn_rprt(self):
		print("\n\nClassification Report: \n{}".format(classification_report(self.y_test.ravel(), np.array(self.y_pred_all, dtype='float'))))


	def plot_piePercentChart(self):
		labels= [r'<5%', r'5%-10%', r'10%-20%', r'20%-50%', r'>50%']
		sizes= [int(round(self.count_0_5_percentage, 0)), int(round(self.count_5_10_percentage, 0)), int(round(self.count_10_20_percentage, 0)), int(round(self.count_20_50_percentage, 0)), int(round(self.count_50_inf_percentage, 0))]
		explode= (0, 0, 0, 0, 0)

		fig1, ax1= plt.subplots()

		ax1.pie(sizes, explode= explode, labels= labels, autopct= '%1.1f%%', shadow= False, startangle= 90)
		ax1.axis('equal')

		plt.show()


	def plot_bucketPercent_stackedBarchart(self):
		p1= plt.bar(0, self.count_0_5_percentage, 0.2, color= 'blue')
		p2= plt.bar(0, self.count_5_10_percentage, 0.2, color= 'green', bottom= self.count_0_5_percentage)
		p3= plt.bar(0, self.count_10_20_percentage, 0.2, color= 'red', bottom= self.count_5_10_percentage+self.count_0_5_percentage)
		p4= plt.bar(0, self.count_20_50_percentage, 0.2, color= 'purple', bottom= self.count_5_10_percentage+self.count_0_5_percentage+self.count_10_20_percentage)
		p5= plt.bar(0, self.count_50_inf_percentage, 0.2, color= 'gray', bottom= self.count_5_10_percentage+self.count_0_5_percentage+self.count_10_20_percentage+self.count_20_50_percentage)
		# plt.title(r'Stacked Barchart of %diff between True and Predicted values')
		plt.xticks([0], ('M1'))
		# plt.xlabel('Model 1')
		plt.ylabel('Relative Error(%)')

		# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('0-5'+' Count: '+str(count_0_5)+'('+str(self.count_0_5_percentage)+'%)'+' out of '+str(self.count), '5-10'+' Count: '+str(count_5_10)+'('+str(self.count_5_10_percentage)+'%)'+' Cum: '+str(count_0_5+count_5_10), '10-20'+' Count: '+str(count_10_20)+'('+str(self.count_10_20_percentage)+'%)'+' Cum: '+str(count_0_5+count_5_10+count_10_20), '20-'+str(self.count)+' Count: '+str(count_20_inf)+'('+str(self.count_20_inf_percentage)+'%)'+' Cum: '+str(self.count)))
		plt.legend((p5[0], p4[0], p3[0], p2[0], p1[0]), (r'>50%: '+str(int(round(self.count_50_inf_percentage))), r'20%-50%: '+str(int(round(self.count_20_50_percentage))), r'10%-20%: '+str(int(round(self.count_10_20_percentage))), r'5%-10%: '+str(int(round(self.count_5_10_percentage))), r'<5%: '+str(int(round(self.count_0_5_percentage)))))

		# plt.clf()

		# self.count_percentage_df= pd.DataFrame({'count': self.count, 'count_0_5_percentage': self.count_0_5_percentage, 'count_5_10_percentage': self.count_5_10_percentage, 'count_10_20_percentage': self.count_10_20_percentage, 'count_20_inf_percentage': self.count_20_inf_percentage})


	def eval_test_regression(self, plot_eval= False):

		##### Marking calculations for count and count_percentages for plotting "plot_bucketPercent_stackedBarchart" later #####

		# Total count
		self.count= self.true_predicted_df.shape[0]

		# Count of values between 0 and 5
		self.true_values_0_5= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(0, 5)]['True'].values
		self.pred_values_0_5= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(0, 5)]['Predicted'].values
		count_0_5= self.true_predicted_df[self.true_predicted_df['%diff'].between(0, 5)].describe().iloc[0:1, 1:2].values[0][0]
		self.count_0_5_percentage= np.round((count_0_5/self.count)*100, 2)

		# Count of values between 5 and 10
		self.true_values_5_10= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(5, 10)]['True'].values
		self.pred_values_5_10= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(5, 10)]['Predicted'].values
		count_5_10= self.true_predicted_df[self.true_predicted_df['%diff'].between(5, 10)].describe().iloc[0:1, 1:2].values[0][0]
		self.count_5_10_percentage= np.round((count_5_10/self.count)*100, 2)

		# Count of values between 10 and 20
		self.true_values_10_20= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(10, 20)]['True'].values
		self.pred_values_10_20= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(10, 20)]['Predicted'].values
		count_10_20= self.true_predicted_df[self.true_predicted_df['%diff'].between(10, 20)].describe().iloc[0:1, 1:2].values[0][0]
		self.count_10_20_percentage= np.round((count_10_20/self.count)*100, 2)

		# Count of values between 20 and 50
		self.true_values_20_50= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(20, 50)]['True'].values
		self.pred_values_20_50= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(20, 50)]['Predicted'].values
		count_20_50= self.true_predicted_df[self.true_predicted_df['%diff'].between(20, 50)].describe().iloc[0:1, 1:2].values[0][0]
		self.count_20_50_percentage= np.round((count_20_50/self.count)*100, 2)

		# Count of values beyond 50
		self.true_values_50_inf= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(50, 10000)]['True'].values
		self.pred_values_50_inf= self.true_predicted_df[self.true_predicted_df[r'%diff'].between(50, 10000)]['Predicted'].values
		count_50_inf= self.true_predicted_df[self.true_predicted_df['%diff'].between(50, 10000)].describe().iloc[0:1, 1:2].values[0][0]
		self.count_50_inf_percentage= np.round((count_50_inf/self.count)*100, 2)


		if plot_eval:
			plt.plot(self.true_values, color= 'green', label= 'True values')
			plt.plot(self.predicted_values, color= 'red', label= 'Predicted values')
			plt.title('True and Predicted values (Regression)')
			plt.legend()

			plt.show()
			plt.clf()

			lim= (0, 160)
			
			plt.plot(self.true_values_50_inf, self.pred_values_50_inf, 'o', label= r'>50%', color= 'gray', markersize= 3)
			plt.plot(self.true_values_20_50, self.pred_values_20_50, 'o', label= r'20%-50%', color= 'purple', markersize= 3)
			plt.plot(self.true_values_10_20, self.pred_values_10_20, 'o', label= r'10%-20%', color= 'red', markersize= 3)
			plt.plot(self.true_values_5_10, self.pred_values_5_10, 'o', label= r'5%-10%', color= 'green', markersize= 3)
			plt.plot(self.true_values_0_5, self.pred_values_0_5, 'o', label= r'<5%', color= 'blue', markersize= 3)
			plt.plot(np.append(self.true_values, [155]), np.append(self.true_values, [155]), label= 'True values', color= 'cyan', lw= 2)

			plt.title("True vs Predicted")
			plt.xlim(lim)
			plt.ylim(lim)
			plt.xlabel(self.labels[0])
			plt.ylabel(self.labels[0])
			plt.grid()
			plt.legend()
			plt.show()

			plt.clf()

		print("\nR squared score: {}".format(r2_score(self.true_values, self.predicted_values)))


	def eval_test_classification(self):
		# self.y_pred= [1.0 if i>= self.thresh else 0.0 for i in self.y_score.ravel()]
		# self.y_pred= np.array(self.y_pred)
		self.y_test= self.y_test.astype('float64')
		self.y_pred_all= [1.0 if i>= self.thresh else 0.0 for i in self.y_score.ravel()]

		self.y_pred= []
		for i in range(len(self.labels)):
			self.y_pred.append(np.array([1.0 if i>=self.thresh else 0.0 for i in self.y_score[:,i]]))

		self.classfctn_rprt()

		# Confusion matrix
		print("\n\nConfusion matrix:\n")
		print(pd.DataFrame(confusion_matrix(self.y_test.ravel(), self.y_pred_all), columns= ['Passed', 'Failed'], index= ['Passed', 'Failed']))

		print("\n\nConfusion Matrices per class:")
		for i in range(len(self.labels)):
			print("\n Class: {}\n".format(self.labels[i]))
			print(pd.DataFrame(confusion_matrix(self.y_test[:,i], self.y_pred[i]), columns= ['Passed', 'Failed'], index= ['Passed', 'Failed']))
			
		# Distribution of predictions
		above_thresh= self.predicted_values[self.predicted_values >= self.thresh]
		below_thresh= self.predicted_values[self.predicted_values < self.thresh]

		plt.hist(above_thresh, color= 'green', label= 'Positives')
		plt.hist(below_thresh, color= 'orange', label= 'Negatives')
		plt.title("Distribution of Predictions (Classification)")
		plt.xlabel("Probabilities")
		plt.ylabel("Count")
		plt.legend(loc= "upper center")
		plt.show()

		plt.clf()

		# Distributions of Actuals
		y_tst= self.y_test.ravel().astype('float')
		above_thresh= y_tst[y_tst == 1.0]
		below_thresh= y_tst[y_tst == 0.0]

		plt.hist(above_thresh, color= 'green', label= 'Positives')
		plt.hist(below_thresh, color= 'orange', label= 'Negatives')
		plt.title("Distribution of Actuals (Classification)")
		plt.xlabel("Classes")
		plt.ylabel("Count")
		plt.legend(loc= "upper center")
		plt.show()

		plt.clf()
		


	def eval_test(self, plot_eval= True):
		# if self.was_normalized:
			# self.X_train= ((self.X_train-self.Xtrain.min())/(self.X_train.max()-self.X_train.min()))

		# print(self.model.evaluate(self.X_test, self.y_test))

		self.predicted_values= self.model.predict(self.X_test).flatten()
		self.true_values= self.y_test.flatten()

		self.true_predicted_df= pd.DataFrame({'True': self.true_values, 'Predicted': self.predicted_values}, index= np.arange(0, len(self.true_values)))
		self.true_predicted_df[r'%diff']= (np.absolute(self.true_values-self.predicted_values)/self.true_values)*100
		print("\n{}".format(self.true_predicted_df.head(10)))


		if self.loss == 'mse' or self.loss == 'mae':
			self.eval_test_regression(plot_eval)
		else:
			self.eval_test_classification()



	def plot_results(self, macro_roc= False, micro_roc= False):
		self.history_dict= self.history.history

		self.plot_loss()

		for i in self.metrics:
			if i == 'acc' or i == "accuracy":
				self.plot_acc()
				# self.roc_crv(macro_roc= macro_roc, micro_roc= micro_roc)
				# self.conf_matrx()
			elif i == 'mae':
				self.plot_mae()
			elif i == 'mse':
				self.plot_mse()
			else:
				pass

	def return_network(self):
		return self.model


	def cross_val(self):
		if 'acc' in self.metrics or 'accuracy' in self.metrics:
			network= KerasClassifier(build_fn= self.return_network,
									epochs= self.epochs,
									batch_size= self.batch_size)
			print("Accuracy values: ", cross_val_score(network, X= self.df.drop(self.labels, axis= 1).values, y= self.df[self.labels].values, cv= self.k))

		else:
			network= KerasRegressor(build_fn= self.return_network,
									epochs= self.epochs,
									batch_size= self.batch_size)

			rsq= make_scorer(r2_score)
			print("R squared values: ",cross_val_score(network, X= self.df.drop(self.labels, axis= 1).values, y= self.df[self.labels].values, cv= self.k, scoring= rsq))


	def l1_l2_reg(self):
		if self.l1 == None and self.l2 == None:
			for i in range(self.n_layers):
				yield None

		elif self.l1 == None and self.l2 != None:
			for i in self.l2:
				if i == None or i == 0:
					yield None
				else:
					yield regularizers.l2(i)

		elif self.l1 != None and self.l2 == None:
			for i in self.l1:
				if i == None or i == 0:
					yield None
				else:
					yield regularizers.l1(i)

		else:
			regs= np.stack((self.l1, self.l2), axis=1)
			for i, j in regs:
				yield regularizers.l1_l2(l1= i, l2= j)


	def n_nodes(self):
		for i in self.n_nodes_per_layer:
			yield i

	def actvn_lyr(self):
		for i in self.actvn_per_layer:
			yield i

	def get_dropout(self):
		if self.dropout == None or self.dropout == 0:
			for i in range(self.n_layers-1):
				yield None
		else:
			for i in self.dropout:
				if i == 0:
					yield None
				else:
					yield i


	def build(self):
		l1l2= self.l1_l2_reg()
		nodes= self.n_nodes()
		atvn= self.actvn_lyr()
		drpt= self.get_dropout()

		# np.random.seed(seed= 42)

		self.train_test_dataset()

		self.model= models.Sequential()

		for n in range(self.n_layers):
			if n == 0:
				self.model.add(layers.Dense(next(nodes), activation= next(atvn), input_shape= (self.X_train.shape[1], ), kernel_initializer= initializers.glorot_uniform(seed= 42), kernel_regularizer= next(l1l2)))
				dpout= next(drpt)
				if dpout != None:
					self.model.add(Dropout(dpout))
			
			elif n+1 == self.n_layers:
				self.model.add(layers.Dense(next(nodes), activation= next(atvn), kernel_regularizer= next(l1l2)))

			else:
				self.model.add(layers.Dense(next(nodes), activation= next(atvn), kernel_regularizer= next(l1l2)))
				dpout= next(drpt)
				if dpout != None:
					self.model.add(Dropout(dpout))
				
		self.model.compile(optimizer= self.set_optimizer(),
						loss= self.loss,
						metrics= self.metrics)


	def train(self, plot_results= False, evaluate_test= False, macro_roc= False, micro_roc= False, cross_val= False):
		self.build()

		if cross_val:
			self.cross_val()

			if plot_results:
				self.plot_results(macro_roc= macro_roc, micro_roc= micro_roc)

			if evaluate_test:
				self.eval_test()

		else:
			np.random.seed(seed= 42)
			self.history= self.model.fit(self.X_train,
								self.y_train,
								epochs= self.epochs,
								batch_size= self.batch_size,
								validation_data= (self.X_val, self.y_val),
								shuffle= False)

			self.y_score= self.model.predict(self.X_test)

			if plot_results:
				self.plot_results(macro_roc= macro_roc, micro_roc= micro_roc)

			if evaluate_test:
				self.eval_test()