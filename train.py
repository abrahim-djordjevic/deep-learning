
#import MNIST data
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)

#import libraries
import tensorflow as tf
import hy_param
import model

#tf graph input
#feed raw images into the model
X = model.X
#feed the labels associated with the images
Y = model.Y

#creating checkpoints which are basically the intermediate steps in training
checkpoint_dir = os.path.abspath(os.path.join(hy_param.checkpoint_dir,
"checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# We only keep the last 2 checkpoints to manage storage
saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
loss = tf.Variable(0.0)
#intialise the variable
init = tf.global_variables_initializer()
all_loss = []

#start training
with tf.Session() as sess:
    writer_1 = tf.summary.FileWriter("./runs/summary/",sess.graph)
    sum_var = tf.summary.scalar("loss", model.accuracy)
    write_op = tf.summary.merge_all()

    #run the initializer
    sess.run(init)
    for step in range(1, hy_param.iterations+1):
        #extracting
        batch_x, batch_y = mnist.train.next_batch(hy_param.batch_size)
        # Run optimization op (backprop)
        sess.run(model.train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % hy_param.display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc, summary = sess.run([model.loss_op, model.accuracy,write_op], feed_dict = {X: batch_x, Y: batch_y})
            all_loss.append(loss)
            writer_1.add_summary(summary, step)
            print("Step " + str(step) + ", Minibatch Loss- "+ \
                "{:.4f}".format(loss) + ", Training Accuracy= " + \
                "{:.3f}".format(acc))
        if step % hy_param.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=step)
                print("Saved model checkpoint to {}\n".format(path))
        print("Testing Accuracy:", sess.run(model.accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels}))
print("optimization Finished")
