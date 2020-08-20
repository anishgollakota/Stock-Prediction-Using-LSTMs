import math
import tensorflow as tf
import pandas_datareader as web
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from preprocessing import x_train
from preprocessing import y_train
from preprocessing import scaled_data
from preprocessing import training_data_len, dataset, scaler

#Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model using an adam optimizer and a MSE (Mean Squared Error) loss function
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=10)

#Create testing dataset
#New array containing scaled values
test_data = scaled_data[training_data_len-60:, :]
#Create datasets x_test, y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape((x_test), (x_test.shape[0], x_test.shape[1], 1))

#Get model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test)**2)

