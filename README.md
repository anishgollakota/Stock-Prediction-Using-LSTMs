# Stock-Prediction-Using-LSTMs
Built a stock prediction program in which the user can pick any company available on Yahoo! Finances and be able to predict the closing price of the stock with a very high accuracy.

## Preprocessing
- Seperation of the data over the last 60 days to use as the testing set and previous values used in the training set.
- Lets user choose the company, start date, and end date for the analytics that they want to predict on

## The Neural Network
The Neural Network in use for this model is an Sequential LSTM model.

The model is comprised of:
- Two LSTM Layers
  - Each consisting of 50 Neurons
  - Do not return sequences on the last layer because we want output predicting the next day's value on the dataset.
- Two Dense Layers
  - Conisting of 25 Neurons and 1 Neuron
- Compilation
  - Optimizer: Adam
  - Loss: MSE (Mean Squared Error)
 
## Example: Predicting NVIDIA stock prices with high accuracy

![](src/NVIDIA%20Closing%20Prices.png)

As you can see, the NVIDIA stock price at the end of the cycle starts to become a little unpredictable.

After applying our model and predictions:

![](src/NVIDIA%20Predicted%20Closing%20Prices.png)

The close correlation between the validations and predictions indicate a reliable model for making business decisions.

