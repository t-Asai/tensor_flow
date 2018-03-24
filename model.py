import tensorflow as tf
import methods

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([783, 10]))
b = tf.Variable(tf.zeros([10]))

W_conv1 = methods.weight_variable([5, 5, 1, 32])
b_conv1 = methods.bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(methods.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = methods.max_pool_2x2(h_conv1)

W_conv2 = methods.weight_variable([5, 5, 32, 64])
b_conv2 = methods.bias_variable([64])

h_conv2 = tf.nn.relu(methods.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = methods.max_pool_2x2(h_conv2)

W_fc1 = methods.weight_variable([7 * 7 * 64, 1024])
b_fc1 = methods.bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = methods.weight_variable([1024, 10])
b_fc2 = methods.bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2