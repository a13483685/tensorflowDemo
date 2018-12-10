import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#随机生成1000个点，围绕在y=0.1x+b
num_points = 1000
voctors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0,0.55)
    y1 = x1*0.1 + 0.3 + np.random.normal(0.0,0.03)
    voctors_set.append([x1,y1])

x_data = [v[0] for v in voctors_set]
y_data = [v[1] for v in voctors_set]

plt.scatter(x_data,y_data,c='r')
plt.show()

W = tf.Variable(tf.random_uniform([1],-1.0,1.0),name='W')
# 生成1维矩阵，初始值是0
b = tf.Variable(tf.zeros([1]),name='b')
# 得到预估值y
y = W*x_data+b

# 以预估值和实际值之间的均方差作为损失
loss = tf.reduce_mean(tf.square(y - y_data),name='loss')
# 采用梯度下降来优化
optimaizer = tf.train.GradientDescentOptimizer(0.5)
# 训练的过程就是最小化误差值
train = optimaizer.minimize(loss,name='train')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print("demo")
print("W = ",sess.run(W),"b = ",sess.run(b),"loss = ",sess.run(loss))

for step in range(20):
    sess.run(train)
    print("W = " , sess.run(W),"b = " , sess.run(b),"loss = " ,sess.run(loss))