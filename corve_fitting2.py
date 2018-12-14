import numpy as np
x = np.random.rand(100).astype(np.float32)
y = 3*x*x+1

import matplotlib.pyplot as plt
plt.plot(x,y,'ro')
plt.legend()
plt.show()

import tensorflow as tf
w1 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
w2 = tf.Variable(tf.random_uniform([1],-1.0,1.0))

w3 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y_ = w1*x+w2*x*x+w3*x*x*x+b

loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for step in range(100):
   sess.run(train)

print(step, sess.run(w1),sess.run(w2),sess.run(w3),sess.run(b))

plt.plot(x,y,'ro')
y__ = sess.run(w1)*x+sess.run(w2)*x*x+sess.run(w3)*x*x*x+sess.run(b)
plt.plot(x,y__)

# plt.plot(x,0.37588856*x+2.09157467*x*x+0.59096539*x*x*x+0.9672721)
plt.legend()
plt.show()