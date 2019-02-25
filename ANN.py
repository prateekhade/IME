'''
ANN API
Inherits DL
'''

from DL import *

class ANN(DL):

	def __init__(self, df= None, name= "DL", labels= None, validation_size= None, test_size= None, n_layers= None, n_nodes_per_layer= None, actvn_per_layer= None, lr_rate= None, optimizer= None, loss= None, metrics= None, epochs= None, batch_size= None, thresh= 0.5, dropout= None, l1= None, l2= None): # df: dataframe to train on, label: label column/s in the dataset-> this is a list of column names
		super().__init__(df= df, name= name, labels= labels, validation_size= validation_size, test_size= test_size, n_layers= n_layers, n_nodes_per_layer= n_nodes_per_layer, actvn_per_layer= actvn_per_layer, lr_rate= lr_rate, optimizer= optimizer, loss= loss, metrics= metrics, epochs= epochs, batch_size= batch_size, thresh= thresh, dropout= dropout, l1= l1, l2= l2)

