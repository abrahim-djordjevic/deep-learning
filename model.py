#import libraries
import hy_param
import tensorflow as tf

#defining inputs using tf.placeholder
#placeholder alloes for inputs to accept certain input values only
X = tf.placeholder("float",[None, hy_param.num_input], name = 'input_x')
Y = tf.placeholder("float",[None, hy_param.num_classes], name = 'input_y')

#weights and bias
#tf.variable allows us to store and update tensors
weights = {
'h1':tf.Variable(tf.random_normal([hy_param.num_input, hy_param.n_hidden_1])),
'h2':tf.Variable(tf.random_normal([hy_param.n_hidden_1, hy_param.n_hidden_2])),
'h3':tf.Variable(tf.random_normal([hy_param.n_hidden_2, hy_param.n_hidden_3])),
'out':tf.Variable(tf.random_normal([hy_param.n_hidden_3, hy_param.num_classes]))
}

bias = {
 'b1': tf.Variable(tf.random_normal([hy_param.n_hidden_1])),
 'b2': tf.Variable(tf.random_normal([hy_param.n_hidden_2])),
 'b3': tf.Variable(tf.random_normal([hy_param.n_hidden_3])),
 'out': tf.Variable(tf.random_normal([hy_param.num_classes]))
}

#define the logistic repression operation
#hidden layer 1
#tf.matmul is used for matrix multiplication
#tf.add is used to return x + y element wise
layer_1 = tf.add(tf.matmul(X, weights['h1']), bias['b1'])
#hidden layer 2
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), bias['b2'])
#hidden layer 3
layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), bias['b3'])
#output layer
layer_out =  tf.matmul(layer_3, weights['out']) + bias['out']

#softmax operation converts output to a value between 0 and 1
prediction = tf.nn.softmax(layer_out, name='prediction')

#define loss and optimise through stochastic gradient descent
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_out,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=hy_param.learning_rate)
train_op = optimizer.minimize(loss_op)

#prediction and calculate accuracy
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='accuracy')
