#-*- coding: utf-8 -*-
'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784 
""" vector """
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes 
""" vector """

# Create model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))	
""" vector """
b = tf.Variable(tf.zeros([10]))		
""" scala """

# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax 
""" vec*vec+scala => vec, finally [None, 10] scale """

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) # Cross entropy
"""
cross-entropy error에 의해 얻은 None(55000) x 10 벡터를 
10에 대한 축을 기준으로 reduction한다 
	(reduction_indices=1이 위의 뜻. 즉 scale=55000
	reduction_indices=0 인 경우, 55000에 대한 축을 기준으로 reduction. 즉 scale=10
	아무 옵션이 없을 경우, 모든 원소를 다 reduce_sum. 즉 그냥 scala값을 내뱉는다)
scale이 55000(N)인 벡터값이 나오면 그 값들을 reduce_mean으로 평균내어 cost는 scala값을 갖게 된다 

"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):

	# 랜덤하게 train data를 선택하여 학습하는 것을 한 세대로 하여 
	# 여러 세대를 거듭하여 최적화 ( w, b를 찾는다 )

        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		
		# mnist.train.next_batch()에서 batch size만큼 랜덤하게 데이터(사진)를 뽑아 학습
		# 이러한 과정을 total_batch만큼 반복하면서 수행
		# 학습시간을 줄이는 효과, overfitting 방지 효과 
		
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})	# 뽑아낸 batch 값으로 optimizer를 수행하여 w, b 업데이트, x1=사진1의 28x28 데이터, x2=사진2의 28x28 데이터
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
