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

# n_hidden_1 = 256
# n_hidden_2 = 128
n_input = 784
n_output = 10

weights = {
    'wc1':tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),
    'wc2':tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
    'wd1':tf.Variable(tf.random_normal([7*7*128,1024],stddev=0.1)),
    'wd2':tf.Variable(tf.random_normal([1024,n_output],stddev=0.1))
}
biases = {
    'bc1':tf.Variable(tf.random_normal([64],stddev=0.1)),
    'bc2':tf.Variable(tf.random_normal([128],stddev=0.1)),
    'bd1':tf.Variable(tf.random_normal([1024],stddev=0.1)),
    'bd2':tf.Variable(tf.random_normal([n_output],stddev=0.1))
}

def conv_basic(_input,_w,_b,_keepratio):
    _input_r = tf.reshape(_input,shape=[-1,28,28,1])
    _conv1 = tf.nn.conv2d(_input_r,_w['wc1'],strides=[1,1,1,1],padding='SAME')#如果滑动的时候边界不够，那么边界补0
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1,_b['bc1']))#激活函数
    #卷积层的输入和输出差不了多少
    #ksize：窗口大小  第一个维度代表
    #batch_size 1-d:1(一般为1) 2-d:h  2-d:w
    #strides 2-d 3-d 表示w h 的步长都是2
    _pool1 = tf.nn.max_pool(_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #当网络进行训练的时候，不让所有的神经元参与，随机的杀死一些节点
    #_keepratio 保留比例如果想保持60%的神经元，那么填0.6
    _pool_dr1 = tf.nn.dropout(_pool1,_keepratio)
    _conv2 = tf.nn.conv2d(_pool_dr1,_w['wc2'],strides=[1,1,1,1],padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2,_b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2,_keepratio)

    #将当前的输出做一个reshape，转化为一个向量的形式
    _densel = tf.reshape(_pool_dr2,[-1,_w['wd1'].get_shape().as_list()[0]])
    #全连接层也是一个wx+b
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_densel,_w['wd1']),_b['bd1']))
    #更多的时候是在全连接层才加上dropout
    _fc_dr1 = tf.nn.dropout(_fc1,_keepratio)
    _out = tf.add(tf.matmul(_fc_dr1,_w['wd2']),_b['bd2'])
    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _densel,
           'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
           }
    return out
print("CNN READY")

a = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1))
print(a)
a = tf.Print(a, [a], "a: ")
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#用placeHolder来站位x,y

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=_pred))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()
print("GRAPH READY")

sess = tf.Session()
sess.run(init)
training_epochs = 15
batch_size = 16
display_step = 1

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = 10
    for i in range(total_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm,feed_dict={x:batch_xs,y:batch_ys,keepratio:0.7})
        avg_cost +=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.})/total_batch

    if epoch % display_step ==0:
        print("Epoch: %30d/%03d cost:%.9f"%(epoch,training_epochs,avg_cost))
        # train_acc = sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.})
        # print("Training accucy: %.3f"%(train_acc))
print("OPTIMAIZATION FINISHED")