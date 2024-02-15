# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:16:49 2024

@author: yash
"""

import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the trained model
loaded_model = pickle.load(open('C:/Users/yash/Documents/Deeplearning/practise/Chemp/trained_model.sav', 'rb'))

# Load the StandardScaler used during training
ss = pickle.load(open('C:/Users/yash/Documents/Deeplearning/practise/Chemp/standard_scaler.sav', 'rb'))

# Define the input data
input_data=(2,2900.00,498.630,4.66900,4,6)

# Convert input data to a numpy array with dtype=object
input_data_as_numpy_array = np.asarray(input_data, dtype=object)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
print("Reshaped Input Data:\n", input_data_reshaped)

# Standardize the input data using the loaded StandardScaler (fit_transform for the first time)
std_data = ss.transform(input_data_reshaped)
print("Standardized Data:\n", std_data)

# Make a prediction using the loaded model
predict = loaded_model.predict(std_data)
print("Prediction:\n", predict)
