# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:39:52 2020

@author: Hemanth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
#importing the data and converting the datetime to index position and the concerting to date format
data = pd.read_csv('adre.us.csv', index_col= [0])
data.index = pd.to_datetime(data.index).date


# concedring the open value as trining set
training_set = data.iloc[:,0:1].values

#scaling the selected train set

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_scaled = sc.fit_transform(training_set)

#creating data structure with 60 time stamps with 1 output
x_train=[]
y_train=[]



for i in range(60,3000):
    x_train.append(train_scaled[i-60:i,0])
    y_train.append(train_scaled[i,0])
    
#converting the obtained value to array
x_train , y_train = np.array(x_train), np.array(y_train)


#reshaping based on Keras, RECURRENT (Putting in 3 dimention ) as RNN needs this shape of input i.e batchSize, timestamp and order
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

regressor = Sequential()


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(x_train,y_train, epochs=100, batch_size=32)


#getting lastes stock price to compare the predicted data

data_test = pd.read_csv('Test.csv')
real_stock_price = data_test.iloc[:,1:2].values

dataset_total = pd.concat((data['Open'], data_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(data_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []

for i in range (60,201):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_stock_price = regressor.predict(x_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price , color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price , color = 'Blue', label = 'predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show() 