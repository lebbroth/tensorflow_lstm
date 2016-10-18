# tensorflow_lstm

With LSTM_Blaine I am trying to build from a tensorflow LSTM on MNIST idea and use it under a chemical prediction environment.

I searched a lot if there are examples on using LSTMs to predict sequentially some output with many features and got the idea of considering that my 30 minutes previous history of 16 different inputs are none other than a MNIST digit.

I modified a lot in this repo so i can pass CSV files and fixed the "Variable Scope" problems by using the  tf.rnn.BasicLSTMCell and using the tf.Get_variable to fix this problem.

The accuracy is weak as my dataset is small. However the model works fine after i did some modifications into the scope variables and some matrix manipulations with panda, numpy and tensorflow. Please feel free to modify.

I'm new to tensorflow and python and I'm doing this work in chemistry. I thought about the MNIST dataset and how an LSTM can read the MNIST samples by sequences and timesteps. Not so much different from a chemical reaction that has multiple vectors into the past.
