# use tf_gpu virtual environment
#hyperparameters

#parameters
learning_rate = 0.01
iterations = 1000
batch_size = 128
display_step = 1

#network parameters
n_hidden_1 = 300 #300 neurons in 1st hidden layer
n_hidden_2 = 300 #300 neurons in 2nd hidden layer
n_hidden_3 = 30

num_input = 784 #MNIST data input (img size [28,28])
num_classes = 10 #0-9

#training_parameters
checkpoint_every = 100
checkpoint_dir = './runs/'
