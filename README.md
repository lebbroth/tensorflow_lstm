# tensorflow_lstm

With LSTM_Blaine I am trying to build from a tensorflow LSTM on MNIST idea and use it under a chemical prediction environment.

I searched a lot if there are examples on using LSTMs to predict sequentially some output with many features and got the idea of considering that my 30 minutes previous history of 16 different inputs are none other than a MNIST digit.

I modified a lot in this repo so i can pass CSV files and fixed the "Variable Scope" problems by using the  tf.rnn.BasicLSTMCell and using the tf.Get_variable to fix this problem.

Unfortunately i'm still having a "Training accuracy" of around 0.1 which is not normal. I am either afraid that I need to normalize my data or there's something missing.
