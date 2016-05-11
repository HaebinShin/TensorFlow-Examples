#-*- coding: utf-8 -*-
'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf Graph Input
"""placeholder : feed 하는 것, update x"""
X = tf.placeholder("float")	
"""[None, None]"""
Y = tf.placeholder("float")	
"""[None, None] 넣는 값에 따라 scale 변화, 옵션 주면 scale 고정 가능"""

# Create Model

# Set model weights
"""variable    : update O, session run 할 때 바뀌는 값"""
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
activation = tf.add(tf.mul(X, W), b)	
"""그래프 그리기만 할 뿐 계산 x , y=W*X+b"""

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss	
"""mean square error"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient descent
"""cost값을 minimize로 w,b 값을 찾는다"""

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):	# get each scala from vector
            sess.run(optimizer, feed_dict={X: x, Y: y})	# x, y: scala

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y:train_Y})), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})	# train_X, train_Y: vector, 에러차이를 확인
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'


    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83,4.668,8.9,7.91,5.7,8.7,3.1,2.1])
    test_Y = numpy.asarray([1.84,2.273,3.2,2.831,2.92,3.24,1.35,1.03])

    print "Testing... (L2 loss Comparison)"
    testing_cost = sess.run(tf.reduce_sum(tf.pow(activation-Y, 2))/(2*test_X.shape[0]),
                            feed_dict={X: test_X, Y: test_Y}) #same function as cost above
    print "Testing cost=", testing_cost
    print "Absolute l2 loss difference:", abs(training_cost - testing_cost)

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
