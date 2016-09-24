'''
Tensorflow LSTM classification of 16x30 images.
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
import pandas as pd

'''
a Tensorflow LSTM that will sequentially input several lines from each single image 
i.e. The Tensorflow graph will take a flat (1,480) features image as it was done in Multi-layer
perceptron MNIST Tensorflow tutorial, but then reshape it in a sequential manner with 16 features each and 30 time_steps.
'''

blaine = genfromtxt('./Desktop/Blaine_CSV_lstm.csv',delimiter=',')  # CSV transform to array
target = [row[0] for row in blaine]             # 1st column in CSV as the targets
data = blaine[:, 1:481]                          #flat feature vectors
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.05, random_state=42)

f=open('cs-training.csv','w')       #1st split for training
for i,j in enumerate(X_train):
        k=np.append(np.array(y_train[i]),j   )
        f.write(",".join([str(s) for s in k]) + '\n')
f.close()
f=open('cs-testing.csv','w')        #2nd split for test
for i,j in enumerate(X_test):
        k=np.append(np.array(y_test[i]),j   )
        f.write(",".join([str(s) for s in k]) + '\n')
f.close()



new_data = genfromtxt('cs-training.csv',delimiter=',')  # Training data
new_test_data = genfromtxt('cs-testing.csv',delimiter=',')  # Test data

x_train=np.array([ i[1::] for i in new_data])
ss = pd.Series(y_train)     #indexing series needed for later Pandas Dummies one-hot vectors
y_train_onehot = pd.get_dummies(ss)

x_test=np.array([ i[1::] for i in new_test_data])
gg = pd.Series(y_test)
y_test_onehot = pd.get_dummies(gg)


# General Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 15
display_step = 1

# Tensorflow LSTM Network Parameters
n_input = 16 # MNIST data input (img shape: 28*28)
n_steps = 30 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 20 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    with tf.variable_scope('cell_def'): 
        lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)
    
    # Get lstm cell output
    with tf.variable_scope('rnn_def'): 
        outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x = np.split(x_train, 33)
        batch_y = np.split(y_train_onehot, 33)
        for index in range(len(batch_x)):
            ouh1 = batch_x[index]
            ouh2 = batch_y[index]
            # Reshape data to get 28 seq of 28 elements
            ouh1 = np.reshape(ouh1,(batch_size, n_steps, n_input))        
            sess.run(optimizer, feed_dict={x: ouh1, y: ouh2})      # Run optimization op (backprop)
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: ouh1, y: ouh2})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: ouh1, y: ouh2})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
                step += 1
print("Optimization Finished!")
    
   