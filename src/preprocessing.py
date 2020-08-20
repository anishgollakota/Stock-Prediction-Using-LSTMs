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

company = input("What stock do you want to predict on?")
df = web.DataReader(company, data_source='yahoo', start='2014-01-01', end='2020-03-04')

print(df.shape)

plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

#Create a dataframe with values of only the closing price of the stock
data = df.filter(['Close'])
dataset = data.values

#80% of the training set allocated to training data
training_data_len = math.ceil(len(dataset)*0.8)

#Scale the data to values in between 0 and 1
def scale_data(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, scaler

scaled_data, scaler = scale_data(dataset)

#Create the training dataset
training_data = scaled_data[:training_data_len, :]

#Split data into x_train and y_train
x_train = []
y_train = []

#Adding data about the last 60 days
for i in range(60, len(training_data)):
  x_train.append(training_data[i-60:i, 0])
  y_train.append(training_data[i, 0])

#Convert training sets to numpy
x_train, y_train = np.array(x_train), np.array(y_train)

#Need to reshape the dimensions of the training set
#Neural network for LSTMs takes in batch size, timesteps, and dimensionality
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

