# PARI ARIVAZHAGAN, SIVA SUBRAMANIAN
# 1001-644-268
# 2019-10-07
# Assignment-02-01

import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import math


def display_images(images):
	# This function displays images on a grid.
	# Farhad Kamangar Sept. 2019
	number_of_images=images.shape[0]
	number_of_rows_for_subplot=int(np.sqrt(number_of_images))
	number_of_columns_for_subplot=int(np.ceil(number_of_images/number_of_rows_for_subplot))
	for k in range(number_of_images):
		plt.subplot(number_of_rows_for_subplot,number_of_columns_for_subplot,k+1)
		plt.imshow(images[k], cmap=plt.get_cmap('gray'))
		# plt.imshow(images[k], cmap=pyplot.get_cmap('gray'))
	plt.show()

def display_numpy_array_as_table(input_array):
	# This function displays a 1d or 2d numpy array (matrix).
	# Farhad Kamangar Sept. 2019
	if input_array.ndim==1:
		num_of_columns,=input_array.shape
		temp_matrix=input_array.reshape((1, num_of_columns))
	elif input_array.ndim>2:
		print("Input matrix dimension is greater than 2. Can not display as table")
		return
	else:
		temp_matrix=input_array
	number_of_rows,num_of_columns = temp_matrix.shape
	plt.figure()
	tb = plt.table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
	for cell in tb.properties()['child_artists']:
		cell.set_height(1/number_of_rows)
		cell.set_width(1/num_of_columns)

	ax = plt.gca()
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()
class Hebbian(object):
	def __init__(self, input_dimensions=2,number_of_classes=4,transfer_function="Hard_limit",seed=None):
		"""
		Initialize Perceptron model
		:param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
		:param number_of_classes: The number of classes.
		:param transfer_function: Transfer function for each neuron. Possible values are:
		"Hard_limit" ,  "Sigmoid", "Linear".
		:param seed: Random number generator seed.
		"""
		if seed != None:
			np.random.seed(seed)
		self.input_dimensions = input_dimensions
		self.number_of_classes=number_of_classes
		self.transfer_function=transfer_function
		self._initialize_weights()
	def _initialize_weights(self):
		"""
		Initialize the weights, initalize using random numbers.
		Note that number of neurons in the model is equal to the number of classes
		"""
		self.weights = []
		self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1)

	def initialize_all_weights_to_zeros(self):
		"""
		Initialize the weights, initalize using random numbers.
		"""
		self.weights = []
		self.weights = np.zeros((self.number_of_classes,self.input_dimensions+1))
		return self.weights

	def predict(self, X):
		"""
		Make a prediction on an array of inputs
		:param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
		as the first row.
		:return: Array of model outputs [number_of_classes ,n_samples]. This array is a numerical array.
		"""
		# print("X is: ", X)
		X=np.insert(X,0,1,axis=0)
		# print(X.shape)
		#print("weights:", self.weights)
		net=np.dot(self.weights,X)
		# print("net",net)
		if self.transfer_function=="Hard_limit":
			for x in range(0,len(net)):
				for y in range(0,len(net[x])):
					if net[x][y]>=0:
						net[x][y]=1
					else:
						net[x][y]=0
			# print("wp:", net)
			return net

		elif self.transfer_function=="Linear":
			return net

		elif self.transfer_function=="Sigmoid":
			return 1/(1+np.exp(-net))


	def print_weights(self):
		"""
		This function prints the weight matrix (Bias is included in the weight matrix).
		"""
	def train(self, X, y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):
		"""
		Given a batch of data, and the necessary hyperparameters,
		this function adjusts the self.weights using Perceptron learning rule.
		Training should be repeted num_epochs time.
		:param X: Array of input [input_dimensions,n_samples]
		:param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
		the desired (true) class.
		:param batch_size: number of samples in a batch
		:param num_epochs: Number of times training should be repeated over all input data
		:param alpha: Learning rate
		:param gamma: Controls the decay
		:param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
		:return: None
		"""
		if learning=="Delta":
			# num_epochs = 2
			for x in range(1,num_epochs):
					
				cl=[]
				limit=math.floor(X.shape[1]/batch_size)
				for limvar in range(1,limit+1):
					cl.append(batch_size*(limvar))

				#print(cl)

				

				#print(splitvar)
				onehot_encoded = list()

				for value in y:
					letter = [0 for _ in range(self.number_of_classes)]
					# print("letter:", letter)
					letter[value] = 1
					# print(letter)
					onehot_encoded.append(np.asarray(letter))
				
				splitvarX=np.split(X,cl,axis=1)
				oharray=np.asarray(onehot_encoded)
				# print("onehot_encoded", oharray)
				#print("splitvarX", splitvarX)

				splitvarY=np.split(oharray,cl,axis=0)
				# print("splitvarY", splitvarY)
				#print("splitvarY", splitvarY)

				for spl in range(0,len(splitvarX)):
					z=self.predict(splitvarX[spl])
					# print("z is:", z)
					#e=y-z
					#print(y)
					#print(z)
					#print(e)
					splitvarX[spl]=np.insert(splitvarX[spl],0,1,axis=0)
					
					e=splitvarY[spl].T-z
					# print("e:", e)

					

					temp=np.dot(e,splitvarX[spl].T)*alpha
					self.weights=self.weights+temp

			
			#print(self.weights)

		elif learning=="Filtered":
			for x in range(1,num_epochs):
					
				cl=[]
				limit=math.floor(X.shape[1]/batch_size)
				for x in range(1,limit+1):
					cl.append(batch_size*(x))

				#print(cl)

				splitvarX=np.split(X,cl,axis=1)

				#print(splitvar)
				onehot_encoded = list()

				for value in y:
					letter = [0 for _ in range(self.number_of_classes)]
					letter[value] = 1
					onehot_encoded.append(np.asarray(letter))
					#print(onehot_encoded)

				oharray=np.asarray(onehot_encoded)

				splitvarY=np.split(oharray,cl,axis=0)
				#print(splitvarY[0].shape)

				for x in range(0,len(splitvarX)):

					#print(splitvarX[x].shape)

					z=self.predict(splitvarX[x])
					#print(z)
					#e=y-z
					#print(y)
					#print(z)
					#print(e)
					
					e=splitvarY[x].T
					#print(e)

					splitvarX[x]=np.insert(splitvarX[x],0,1,axis=0)
					temp=np.dot(e,splitvarX[x].T)*alpha
					self.weights=((1-gamma)*self.weights)+temp

			
			#print(self.weights)

		elif learning=="Unsupervised_hebb":
			for x in range(1,num_epochs):
					
				cl=[]
				limit=math.floor(X.shape[1]/batch_size)
				for x in range(1,limit+1):
					cl.append(batch_size*(x))

				#print(cl)

				splitvarX=np.split(X,cl,axis=1)

				#print(splitvar)
				onehot_encoded = list()

				for value in y:
					letter = [0 for _ in range(self.number_of_classes)]
					letter[value] = 1
					onehot_encoded.append(np.asarray(letter))
					#print(onehot_encoded)

				oharray=np.asarray(onehot_encoded)

				splitvarY=np.split(oharray,cl,axis=0)
				#print(splitvarY[0].shape)

				for x in range(0,len(splitvarX)):

					#print(splitvarX[x].shape)

					e=self.predict(splitvarX[x])
					#print(z)
					#e=y-z
					#print(y)
					#print(z)
					#print(e)
					
					
					#print(e)

					splitvarX[x]=np.insert(splitvarX[x],0,1,axis=0)
					temp=np.dot(e,splitvarX[x].T)*alpha
					self.weights=self.weights+temp

			
			#print(self.weights)








	def calculate_percent_error(self,X, y):
		"""
		Given a batch of data this function calculates percent error.
		For each input sample, if the predicted class output is not the same as the desired class,
		then it is considered one error. Percent error is number_of_errors/ number_of_samples.
		:param X: Array of input [input_dimensions,n_samples]
		:param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
		the desired (true) class.
		:return percent_error
		"""
		tempctr=0
		tempPred=self.predict(X)
		for x in range(1,tempPred.shape[1]):
			if np.argmax(tempPred[:,x])!=y[x]:
				tempctr+=1
		
		return tempctr/X.shape[1]





	def calculate_confusion_matrix(self,X,y):
		"""
		Given a desired (true) output as one hot and the predicted output as one-hot,
		this method calculates the confusion matrix.
		If the predicted class output is not the same as the desired output,
		then it is considered one error.
		:param X: Array of input [input_dimensions,n_samples]
		:param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
		the desired (true) class.
		:return confusion_matrix[number_of_classes,number_of_classes].
		Confusion matrix should be shown as the number of times that
		an image of class n is classified as class m where 1<=n,m<=number_of_classes.
		"""
		tempPred=self.predict(X)
		CM=np.zeros((self.number_of_classes,self.number_of_classes))
		for l,m in enumerate(y):
			CI=np.argmax(tempPred[:,l])
			CM[m][CI]=CM[m][CI]+1

		return CM




if __name__ == "__main__":

	# Read mnist data
	number_of_classes = 10
	number_of_training_samples_to_use = 700
	number_of_test_samples_to_use = 100
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	X_train_vectorized=((X_train.reshape(X_train.shape[0],-1)).T)[:,0:number_of_training_samples_to_use]
	# print(type(X_train_vectorized))
	#print(X_train_vectorized.shape)
	y_train = y_train[0:number_of_training_samples_to_use]
	#print(y_train.shape)
	X_test_vectorized=((X_test.reshape(X_test.shape[0],-1)).T)[:,0:number_of_test_samples_to_use]
	#print(X_test_vectorized.shape)
	y_test = y_test[0:number_of_test_samples_to_use]
	#print(y_test.shape)
	number_of_images_to_view=16
	test_x=X_train_vectorized[:,0:number_of_images_to_view].T.reshape((number_of_images_to_view,28,28))
	#display_images(test_x.shape)
	input_dimensions=X_test_vectorized.shape[0]
	model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
					transfer_function="Hard_limit",seed=5)
	# model.initialize_all_weights_to_zeros()
	percent_error=[]
	for k in range (10):
		model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Delta")
		percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
	print("******  Percent Error ******\n",percent_error)
	confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
	print(np.array2string(confusion_matrix, separator=","))
