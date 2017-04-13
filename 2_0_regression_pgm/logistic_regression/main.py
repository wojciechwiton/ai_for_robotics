#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:13:01 2017

@author: sebastian
"""
import numpy as np
import matplotlib.pyplot as plt


def logistic_function(w, x):
    # TODO implement the logistic function
    fun = 1 / (1 + np.exp(-np.dot(w.transpose(), x.transpose())))
    return fun # change this line

def calculate_grad(w, x, y, lam):
	n_samples = x.shape[0]
	n_features = x.shape[1]
	grad = np.zeros((n_features, 1))
	y_temp = y.reshape((1,len(y)))

	w[0] = 0

	for n in range(n_features):
		grad[n] = np.sum((np.dot(logistic_function(w, x) - y_temp, x[:,n]) + lam * w[n]),axis=0) / n_samples

	# for n in range(n_features):
	# 	sum = 0
	# 	for i in range(n_samples):
	# 		xi = x[i].reshape(1, len(x[i]))
	# 		sum += (logistic_function(w, xi) - y[i]) * x[i, n]

	# 	grad[n] = (sum + lam * w[n]) / n_samples

	return grad

def calculate_H(w, x, lam):
	n_samples = x.shape[0]
	n_features = x.shape[1]
	H = np.zeros((n_features, n_features))
	for i in range(n_samples):
		xi = x[i].reshape(1, len(x[i]))
		H += logistic_function(w, xi) * (1 - logistic_function(w, xi)) * np.dot(xi.transpose(), xi)
	
	temp = np.identity(n_features)
	temp[0,0] = 0
	H = (H + np.dot(lam, temp)) / n_samples

	# print(H)
	return H

# To make it easier the 24x24 pixels have been reshaped to a vector of 576 pixels. the value corrsponds to the greyscale intensity of the pixel
input_data = np.genfromtxt(open("XtrainIMG.txt"))  # This is an array that has the features (all 576 pixel intensities) in the columns and all the available pictures in the rows
output_data = np.genfromtxt(open("Ytrain.txt"))  # This is a vector that has the classification (1 for open eye 0 for closed eye) in the rows

n_samples = input_data.shape[0]
n_features = input_data.shape[1]

input_data = np.hstack((np.ones((n_samples, 1)), input_data))

ratio_train_validate = 0.8
idx_switch = int(n_samples * ratio_train_validate)
training_input = input_data[:idx_switch, :]
training_output = output_data[:idx_switch][:,None]
validation_input = input_data[idx_switch:, :]
validation_output = output_data[idx_switch:][:,None]

#TODO initialise w
w = np.zeros((n_features + 1, 1)) # change this line

#TODO implement the iterative calculation of w
for i in range(10):
	print(i)
	grad = calculate_grad(w, training_input, training_output, 0.5)
	H = calculate_H(w, training_input, 0.5)
	H_inv = np.linalg.inv(H)
	diff = np.dot(H_inv, grad)
	w_old = w
	w = w - diff
	if np.allclose(w, w_old, 1e-3):
		break
#TODO2: modify the algorithm to account for regularization as well to improve the classifier

#validation
h = logistic_function(w,validation_input)
output = np.round(h).transpose()

error = np.abs(output-validation_output).sum()

print 'wrong classification of ',(error/output.shape[0]*100),'% of the cases in the validation set'


# classify test data for evaluation
test_input = np.genfromtxt(open("XtestIMG.txt"))
n_test_samples = test_input.shape[0]
test_input = np.hstack((np.ones((n_test_samples, 1)), test_input))
h = logistic_function(w,test_input)
test_output = np.round(h).transpose()
np.savetxt('results.txt', test_output)
