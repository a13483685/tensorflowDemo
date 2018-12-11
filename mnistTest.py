# import mnist as mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('data/',one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testing = mnist.test.images
testlabel = mnist.test.labels
print("MNIST loaded")

print(trainimg.shape)
print(trainlabel.shape)
print(testing.shape)
print(testlabel.shape)

n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_class = 10

x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_class])

stddev = 0.1
weight = {
    'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
    'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_class],stddev=stddev))
}

biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_class]))
}
print("network ready")

def multilayer_perceptron(_X,_weight,_biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X,_weight['w1']),_biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,_weight['w2']),_biases['b2']))
    return (tf.matmul(layer_2,_weight['out'])+_biases['out'])

pred = multilayer_perceptron(x,weight,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accr = tf.reduce_mean(tf.cast(corr,"float"))

init = tf.global_variables_initializer()
print("FUNCTIONS READY")

training_epochs = 20
batch_size = 100
display_step = 4
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
  avg_cost =0.
  total_batch = int(mnist.train.num_example/batch_size)
  for i in range(total_batch):
      batch_xs,batch_ys = mnist.train.next_batch(batch_size)
      feeds = {x:batch_xs,y:batch_ys}
      sess.run(optm,feed_dict=feeds)
      avg_cost += sess.run(cost,feed_dict=feeds)
  avg_cost = avg_cost/total_batch
  if(epoch+1)%display_step == 0:
        print("Epoch:%03d/%03d cost : %.9f"%(epoch,training_epochs,avg_cost))
        feeds = {x:batch_xs,y:batch_ys}
        train_acc = sess.run(accr,feed_dict=feeds)
        print("Train acc:%.3f" % (train_acc))
        feeds = {x:mnist.test.images,y:mnist.test.labels}
        test_acc = sess.run(accr,feed_dict=feeds)
        print("test accurary:%.3f"%(test_acc))
print("optimization finished")