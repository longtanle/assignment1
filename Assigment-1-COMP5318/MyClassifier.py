import os
import sys

import numpy as np
from numpy import nan
from numpy import asarray

import pandas as pd

# Libraries for Pre-Processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Libraries for 10-fold cross-validation
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV

# Libraries for classifiers
## K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
## Logistic Regression
from sklearn.linear_model import LogisticRegression
## Naive Bayes
from sklearn.naive_bayes import GaussianNB
## Decision Tree
from sklearn.tree import DecisionTreeClassifier
## Ensembles: Bagging
from sklearn.ensemble import BaggingClassifier
## Ensembles: Ada Boost
from sklearn.ensemble import AdaBoostClassifier
## Ensembles: Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

## Linear SVM
from sklearn.svm import SVC
## Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initialize cvKFold as 10-fold cross-validation
cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Pre Processing Data
def PreProcessing(filename):

	# Read dataset
	dataset = pd.read_csv(filename)

	# Mark missing values as NaN
	dataset = dataset.replace('?', nan)

	# Get attribute data
	dataset_attributes = dataset.iloc[:,0:-1]

	# Replace NaN with the mean value of the column using Simple Impute
	imputer = SimpleImputer(missing_values=nan, strategy='mean')
	attributes = imputer.fit_transform(dataset_attributes)

	# Normalisation attributes using a min-max scaler
	# Define min-max scaler
	scaler = MinMaxScaler()
	# Transform data
	scaled = scaler.fit_transform(attributes)

	print(scaled)

	# Convert attribute data to list
	attribute_list = []
	for i in scaled:
		data = []
		for j in i:
			data.append('%0.4f' % j)
		attribute_list.append(data)


    # Get class list
	class_list = dataset.iloc[:, -1].tolist()

    # Encode classes using Label Encoder
	classes = np.unique(class_list)

	classEncode = LabelEncoder()
	classEncode.fit(classes)

	class_encoder = classEncode.transform(class_list)

	# numClass = len(classes)
	class_encoder=class_encoder.astype(np.float64)

	# Print dataset after pre-processing
	numSamples = len(attribute_list)
	# df = pd.DataFrame(columns=[0:numSamples])
	for i in range(numSamples):
		for j in attribute_list[i]:
			print(j, end = ',')
		if i < numSamples - 1:
			print(int(class_encoder[i]))
		else:
			print(int(class_encoder[i]), end = '')
		# df = df.append([attribute_list[i],class_encoder[i]])

	return attribute_list, class_list

# Read normalised dataset
def readData(filename):

	# Read dataset
	dataset = pd.read_csv(filename)

	# Get attribute data
	dataset_attributes = dataset.iloc[:,0:-1]
	attributes = dataset_attributes.values

	# print(attributes)

	# Convert attribute data to list
	attribute_list = []
	for i in attributes:
		data = []
		for j in i:
			data.append('%0.4f' % j)
		attribute_list.append(data)

    # Get class list
	class_list = dataset.iloc[:, -1].tolist()

    # Encode classes using Label Encoder
	classes = np.unique(class_list)

	classEncode = LabelEncoder()
	classEncode.fit(classes)

	class_encoder = classEncode.transform(class_list)

	# numClass = len(classes)
	class_encoder=class_encoder.astype(np.float64)

	return attribute_list, class_list

# Read parameter files
def readParameter(filename):

	parameter_file = pd.read_csv(filename)

	# Convert parameters to a list:
	parameters = parameter_file.iloc[0].tolist()

	return parameters

# K-Nearest Neighbour
def kNNClassifier(X,y,K):

	neigh = KNeighborsClassifier(n_neighbors=K)

	scores = cross_val_score(neigh, asarray(X, dtype='float64'), y, cv=cvKFold)

	# print("{:.4f}".format(scores.mean()), end='')

	return scores, scores.mean()

# Logistic Regression
def logregClassifier(X, y):

	lr = LogisticRegression(random_state=0)

	scores = cross_val_score(lr, asarray(X, dtype='float64'), y, cv=cvKFold)

	# print("{:.4f}".format(scores.mean()), end='')
	return scores, scores.mean()

# Naive Bayes
def nbClassifier(X, y):

	nb = GaussianNB()

	scores = cross_val_score(nb, asarray(X, dtype='float64'), y, cv=cvKFold)

	# print("{:.4f}".format(scores.mean()), end='')

	return scores, scores.mean()

# Decision Tree
def dtClassifier(X, y):

	dt = DecisionTreeClassifier(criterion='entropy', random_state=0)

	scores = cross_val_score(dt, asarray(X, dtype='float64'), y, cv=cvKFold) 

	# print("{:.4f}".format(scores.mean()), end='')

	return scores, scores.mean()

# Bagging
def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):

	bag = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=0), 
                            n_estimators=n_estimators, max_samples=max_samples, random_state=0)

	scores = cross_val_score(bag, asarray(X, dtype='float64'), y, cv=cvKFold)

	# print("{:.4f}".format(scores.mean()), end='')

	return scores, scores.mean()

# Ada Boost
def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):

	ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth,criterion='entropy', random_state=0), 
							 n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)

	scores = cross_val_score(ada, asarray(X, dtype='float64'), y, cv=cvKFold)

	# print("{:.4f}".format(scores.mean()), end='')

	return scores, scores.mean()

# Gradient Boosting
def gbClassifier(X, y, n_estimators, learning_rate):

	gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)

	scores = cross_val_score(gb, asarray(X, dtype='float64'), y, cv=cvKFold)

	# print("{:.4f}".format(scores.mean()), end='')

	return scores, scores.mean()

# Linear SVM
def bestLinClassifier(X,y):

	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

	param_grid = {
		'C': [0.001, 0.01, 0.1, 1, 10, 100],
		'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
	}

	grid_search = GridSearchCV(SVC(kernel="linear", random_state=0), 
                               param_grid, cv=cvKFold, return_train_score=True)
	grid_search.fit(X_train, y_train)

	# Optimal C value
	print(grid_search.best_params_['C'])
	# Optimal Gamma value
	print(grid_search.best_params_['gamma'])
	# Best accuracy score
	print("{:.4f}".format(grid_search.best_score_))
	# Test set accuracy score
	print("{:.4f}".format(grid_search.score(X_test, y_test)), end='')

# Random Forest
def bestRFClassifier(X,y):

	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

	param_grid = {
		'n_estimators': [10, 30],
		'max_features': ['sqrt'],
		'max_leaf_nodes': [4, 16]
	}

	grid_search = GridSearchCV(RandomForestClassifier(random_state=0, criterion='entropy'), 
                               param_grid, cv=cvKFold, return_train_score=True)
	grid_search.fit(X_train, y_train)

	# Optimal n_estimators
	print(grid_search.best_params_['n_estimators'])
	# Optimal max_leaf_nodes
	print(grid_search.best_params_['max_leaf_nodes'])
	# Best accuracy score
	print("{:.4f}".format(grid_search.best_score_))
	# Test set accuracy score
	print("{:.4f}".format(grid_search.score(X_test, y_test)), end='')

# Main function
if __name__ == '__main__':

	# for idx, arg in enumerate(sys.argv):
	#	print("Argument #{} is {}".format(idx, arg))

	# print ("No. of arguments passed is ", len(sys.argv))

	X,y = readData(sys.argv[1])
	# X,y = PreProcessing(sys.argv[1])

	# Pre Processing Task
	if sys.argv[2] == 'P':

		attribute_list, class_list = PreProcessing(sys.argv[1])	

	# kNN classifier
	if sys.argv[2] == 'NN':

		parameter_list = readParameter(sys.argv[3])
		K = int(parameter_list[0])

		scores, scores_mean = kNNClassifier(X, y, K)

		print("{:.4f}".format(scores.mean()), end='')

	# Linear Regression classifier
	if sys.argv[2] == 'LR':

		scores, scores_mean = logregClassifier(X, y)

		print("{:.4f}".format(scores.mean()), end='')

	# Naive Bayes classifier
	if sys.argv[2] == 'NB':

		scores, scores_mean = nbClassifier(X, y)

		print("{:.4f}".format(scores.mean()), end='')

	# Decision Tree classifier
	if sys.argv[2] == 'DT':

		scores, scores_mean = dtClassifier(X, y)

		print("{:.4f}".format(scores.mean()), end='')

	# Bagging classifier
	if sys.argv[2] == 'BAG':

		parameter_list = readParameter(sys.argv[3])

		n_estimators = int(parameter_list[0])
		max_samples = int(parameter_list[1])
		max_depth = int(parameter_list[2])

		scores, scores_mean = bagDTClassifier(X, y, n_estimators, max_samples, max_depth)

		print("{:.4f}".format(scores.mean()), end='')

	# ADA Boost classifier
	if sys.argv[2] == 'ADA':

		parameter_list = readParameter(sys.argv[3])

		n_estimators = int(parameter_list[0])
		learning_rate = parameter_list[1]
		max_depth = int(parameter_list[2])

		scores, scores_mean = adaDTClassifier(X, y, n_estimators, learning_rate, max_depth)

		print("{:.4f}".format(scores.mean()), end='')		

	# Gradient Boosting classifier
	if sys.argv[2] == 'GB':

		parameter_list = readParameter(sys.argv[3])

		n_estimators = int(parameter_list[0])
		learning_rate = parameter_list[1]

		scores, scores_mean = gbClassifier(X, y, n_estimators, learning_rate)

		print("{:.4f}".format(scores.mean()), end='')		


	# SVM classifier
	if sys.argv[2] == 'SVM':

		bestLinClassifier(X, y)

	# Random Forest classifier
	if sys.argv[2] == 'RF':

		bestRFClassifier(X, y)








