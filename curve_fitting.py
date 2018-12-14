# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xlrd
import matplotlib.pyplot as plt

workbook = xlrd.open_workbook('./data/123.xlsx')
booksheet = workbook.sheet_by_index(1)         #用索引取第一个sheet

row = booksheet.nrows
col = booksheet.ncols

y_data = booksheet.col_values(1)
x_data = booksheet.col_values(2)
print(x_data)
# Graphic display
plt.plot(x_data, y_data, 'ro',label="old")
# x_trian  = np.array(x_data,dtype='f')
# plt.plot(x_trian,1.964*x_trian-111.272,label='new')
plt.legend()
plt.show()
# import tensorflow as tf
#
# # W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# # b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(3.0, dtype=tf.float32)
# b = tf.Variable(1.0, dtype=tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
#
# y = tf.placeholder(tf.float32)
#
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# # optimizer
#
# optimizer = tf.train.GradientDescentOptimizer(1)
# train = optimizer.minimize(loss)
#
# # x_train = [1, 2, 3, 4]
# # y_train = [0, -1, -2, -3]
#
# x_train = np.array(x_data,dtype='f')
# y_train = np.array(y_data,dtype='f')
#
# print(x_train)
# print(y_train)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init) # reset values to wrong
#
# for i in range(82):
#   sess.run(train, {x: x_train, y: y_train})
#
# curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# init = tf.initialize_all_variables()
#
# sess = tf.Session()
# sess.run(init)
#
# for step in range(82):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(W), sess.run(b))
#         print(step, sess.run(loss))
#
#         # Graphic display
#         plt.plot(x_data, y_data, 'ro',label="old")
#         plt.plot(x_data, sess.run(W) * x_data + sess.run(b),label="new")
#         plt.xlabel('x')
#         #plt.xlim(-2, 2)
#         #plt.ylim(0.1, 0.6)
#         plt.ylabel('y')
#         plt.legend()
#         plt.show()
