from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import model

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=model.y_, logits=model.y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(model.y_conv, 1), tf.argmax(model.y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                model.x: batch[0], model.y_: batch[1], model.keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            model.x: mnist.test.images, model.y_: mnist.test.labels, model.keep_prob: 1.0}))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    flag = 1
    for i in range(2):
        batch = mnist.train.next_batch(50)
        # train_step.run(feed_dict={x: batch[0], y_: toOne(batch[1], flag), keep_prob: 0.5})
        train_step.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.5})

        if i % 1000 == 0:
            # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: toOne(batch[1], flag), keep_prob: 1.0})
            train_accuracy = accuracy.eval(
                feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            # print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: toOne(mnist.test.labels, flag), keep_prob: 1.0}))
            print('test accuracy %g' % accuracy.eval(
                feed_dict={model.x: mnist.test.images, model.y_: mnist.test.labels, model.keep_prob: 1.0}))
    print(model.y_conv.eval(feed_dict={model.x: mnist.test.images, model.keep_prob: 1.0}))
    for test_x, test_y in zip(mnist.test.images, mnist.test.labels):
        answer = sess.run(tf.argmax(model.y_conv, 1), feed_dict={
                          model.x: [test_x], model.keep_prob: 1.0})
        if test_y.argmax(0) != answer[0]:
            print(test_y.argmax(0), answer[0])
