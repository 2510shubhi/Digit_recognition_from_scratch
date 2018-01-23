import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from random import randint
import numpy as np
### initialize variables

logs_path = 'log_mnist_4layer'
batch_size = 100
learning_rate = 0.004
training_epochs = 10

 ### Load MNIST dataset

mnist = input_data.read_data_sets("data", one_hot=True)

### Input and Output placeholers

X_input = tf.placeholder(tf.float32, [None, 784], name="input")
Y_output = tf.placeholder(tf.float32, [None, 10])

### Hidden layers

h_l1 = 200  ### hidden_layer number = number of neurons in that layer
h_l2 = 150
h_l3 = 70

### Weights and bias matrix

W1 = tf.Variable(tf.truncated_normal([784, h_l1], stddev = 0.1))
b1 = tf.Variable(tf.ones([h_l1])/10)
W2 = tf.Variable(tf.truncated_normal([h_l1, h_l2], stddev = 0.1))
b2 = tf.Variable(tf.ones([h_l2])/10)
W3 = tf.Variable(tf.truncated_normal([h_l2, h_l3], stddev = 0.1))
b3 = tf.Variable(tf.ones([h_l3])/10)
W4 = tf.Variable(tf.truncated_normal([h_l3, 10], stddev = 0.1))
b4 = tf.Variable(tf.zeros([10]))

### forward and backward propagation

XX = tf.reshape(X_input, [-1, 784])
Y1 = tf.nn.relu(tf.matmul(XX, W1)+b1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2)+b2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3)+b3)
Y_logits=tf.matmul(Y3,W4)+b4
Y=tf.nn.softmax(Y_logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits,labels=Y_output)
cross_entropy=tf.reduce_mean(cross_entropy)*100
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, \
                                    graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, summary = sess.run([train_step, summary_op],\
                                  feed_dict={X_input: batch_x,\
                                             Y_output: batch_y})
            writer.add_summary(summary, epoch * batch_count + i)
        print("Epoch: ", epoch)

    print("Accuracy: ", accuracy.eval(feed_dict={X_input: mnist.test.images, Y_output: mnist.test.labels}))
    print("done")

    num = randint(0, mnist.test.images.shape[0])
    img = mnist.test.images[num]

    classification = sess.run(tf.argmax(Y, 1), feed_dict={X_input: [img]})
    print('Neural Network predicted', classification[0])
    print('Real label is:', np.argmax(mnist.test.labels[num]))
