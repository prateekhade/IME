from DL import *

from keras.models import Model
'''
Remember, X_train's(or X_test's) shape should be (Number of example, height, width, channels), eg.: (60000, 28, 28, 1)
and y_train's(or y_test's) shape should be (Number of examples, ), eg.: (60000, )
'''
class CNN(DL):

	def __init__(self, df= pd.DataFrame(), name= "DL", X_train= None, X_test= None, y_train= None, y_test= None, X_val= None, y_val= None, labels= None, validation_size= None, test_size= None, n_layers= None, n_nodes_per_layer= None, actvn_per_layer= None, lr_rate= None, optimizer= None, loss= None, metrics= None, epochs= None, batch_size= None, thresh= 0.5, dropout= None, l1= None, l2= None, layers_list= None): # df: dataframe to train on, label: label column/s in the dataset-> this is a list of column names
		super().__init__(df= df, name= name, labels= labels, validation_size= validation_size, test_size= test_size, n_layers= n_layers, n_nodes_per_layer= n_nodes_per_layer, actvn_per_layer= actvn_per_layer, lr_rate= lr_rate, optimizer= optimizer, loss= loss, metrics= metrics, epochs= epochs, batch_size= batch_size, thresh= thresh, dropout= dropout, l1= l1, l2= l2)

		self.layers_list= layers_list #List of all the layers in the order they will be stacked
		self.layers= pd.DataFrame(None)
		self.X_train= X_train
		self.X_test= X_test
		self.y_train= y_train
		self.y_test= y_test
		self.X_val= X_val
		self.y_val= y_val
		self.on_ht_ncde= False
		# self.layer_details= layer_details
		# self.details_df= pd.DataFrame(columns= ['filter', 'kernel_size', 'strides', 'padding', ])


	# Getters
	@property
	def layers_list(self):
		return self._layers_list


	# Setters
	@layers_list.setter
	def layers_list(self, n):
		self._layers_list= n


	# APIs
	def img_params(self):
		self.img_rows_ht= self.X_train.shape[1]
		self.img_cols_wd= self.X_train.shape[2]
		self.img_channels= self.X_train.shape[3]


	def lyr_list(self):
		if self.layers_list:
			for i in range(len(self.layers_list)):
				self.layers_list[i]= self.layers_list[i]+'_'+str(i+1)


	def one_hot_encode(self):
		self.on_ht_ncde= True
		self.labels= pd.Series(self.y_train).unique()
		uniq_labs= pd.Series(self.y_train).nunique()
		self.y_train= np.eye(uniq_labs)[self.y_train]
		self.y_test= np.eye(uniq_labs)[self.y_test]


	def parameters(self):
		self.lyr_list()

		self.layers= pd.DataFrame(index= ['filter', 'kernel_size', 'strides', 'padding', 'activation_function', 'l1', 'l2'], columns= self.layers_list)
		print("\nValidation size: {}\nLayers: {}\nLearning rate: {}\nOptimizer: {}\nLoss: {}\nMetrics: {}\nEpochs: {}\nBatch size: {}\n".format(self.validation_size, self.layers_list, self.lr_rate, self.optimizer, self.loss, self.metrics, self.epochs, self.batch_size))
		print("\nCNN model: \n\n", self.layers)


	def roc_crv(self, *args):
		fpr, tpr, thresholds= roc_curve(self.y_test.ravel(), self.y_score.ravel())
		roc_auc= auc(fpr, tpr)

		lw= 2
		plt.figure()
		plt.plot(fpr, tpr, color= 'darkorange', lw=lw, label="ROC curve (area = %0.2f)" %roc_auc)
		plt.plot([0, 1], [0, 1], color= 'navy', lw=lw, linestyle= '--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False positive rate')
		plt.ylabel('True positive rate')
		plt.title('Reciever operating characteristic example')
		plt.legend(loc= 'lower right')
		plt.show()


	def classfctn_rprt(self):
		print("\n\nClassification Report: \n{}".format(classification_report(self.true_values, self.predicted_values)))


	def eval_test_classification(self):
		self.classfctn_rprt()

		# Confusion matrix
		print("\n\nConfusion matrix:\n")
		self.conf_mat= pd.DataFrame(confusion_matrix(self.true_values, self.predicted_values))
		print(self.conf_mat)


	def eval_test(self, plot_eval= True):
		self.predicted_values= []
		self.true_values= []

		for i in range(len(self.X_test)):
			self.predicted_values.append(self.model.predict(self.X_test[i].reshape(1, self.img_rows_ht, self.img_cols_wd, self.img_channels)).argmax())
			self.true_values.append(self.y_test[i].argmax())

		self.true_predicted_df= pd.DataFrame({'True': self.true_values, 'Predicted': self.predicted_values}, index= np.arange(0, len(self.true_values)))
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
				self.roc_crv()
				# self.conf_matrx()
			elif i == 'mae':
				self.plot_mae()
			elif i == 'mse':
				self.plot_mse()
			else:
				pass


	def vizualize_convolutions(self):
		layer_outputs= [layer.output for layer in cnn.model.layers]
		activation_model= Model(inputs= cnn.model.input, outputs= layer_outputs)
		activations = activation_model.predict(cnn.X_train[11].reshape(1,28,28,1))

		print(cnn.model.summary())

		def display_activation(activations, col_size, row_size, act_index): 
		    activation = activations[act_index]
		    activation_index=0
		    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
		    for row in range(0,row_size):
		        for col in range(0,col_size):
		            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
		            activation_index += 1


	def combine_X_y(self):
		self.X= np.vstack([self.X_train, self.X_test])
		if self.on_ht_ncde:
			self.y= np.vstack([self.y_train, self.y_test])
		else:
			self.y= np.hstack([self.y_train, self.y_test])


	def cross_val(self):
		if 'acc' in self.metrics or 'accuracy' in self.metrics:
			network= KerasClassifier(build_fn= self.return_network,
									epochs= self.epochs,
									batch_size= self.batch_size)
			print("Accuracy values: ", cross_val_score(network, X= self.X, y= self.y, cv= self.k))

		else:
			network= KerasRegressor(build_fn= self.return_network,
									epochs= self.epochs,
									batch_size= self.batch_size)

			rsq= make_scorer(r2_score)
			print("R squared values: ",cross_val_score(network, X= self.X, y= self.y, cv= self.k, scoring= rsq))



	def build(self):
		self.combine_X_y()
		self.img_params()
		self.val_dataset()
		self.model= models.Sequential()

		for i in self.layers.columns:
			if 'Conv2D_1' in i:
				self.model.add(layers.Conv2D(self.layers[i]['filter'], self.layers[i]['kernel_size'], strides= self.layers[i]['strides'], padding= self.layers[i]['padding'], activation= self.layers[i]['activation_function'], input_shape= self.X_train.shape[1:]))
			elif 'Conv2D' in i:
				self.model.add(layers.Conv2D(self.layers[i]['filter'], self.layers[i]['kernel_size'], strides= self.layers[i]['strides'], padding= self.layers[i]['padding'], activation= self.layers[i]['activation_function']))
			elif 'MaxPooling2D' in i:
				self.model.add(layers.MaxPooling2D(self.layers[i]['kernel_size']))
			elif 'Flatten' in i:
				self.model.add(layers.Flatten())
			elif 'Dense' in i:
				self.model.add(layers.Dense(self.layers[i]['filter'], activation= self.layers[i]['activation_function']))
	

		self.model.compile(optimizer= self.set_optimizer(),
						loss= self.loss,
						metrics= self.metrics)