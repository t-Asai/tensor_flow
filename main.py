from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from model import x, y_, keep_prob, y_conv
from methods_for_adjust_params import toOne


def run_to_cal_accuracy(data, accuracy, train_step):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = data.train.next_batch(50)
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                # print(batch)
                print('test accuracy %g' % accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



def run_to_estimate(data, accuracy, train_step):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        flag = 1
        for i in range(20000):
            batch = data.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: toOne(
                batch[1], flag), keep_prob: 0.5})
            # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            if i % 1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: toOne(batch[1], flag), keep_prob: 1.0})
                # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                print('test accuracy %g' % accuracy.eval(feed_dict={x: data.test.images, y_: toOne(data.test.labels, flag), keep_prob: 1.0}))
                # print('test accuracy %g' % accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))
        print(y_conv.eval(feed_dict={x: data.test.images, keep_prob: 1.0}))
        for test_x, test_y in zip(data.test.images, data.test.labels):
            answer = sess.run(tf.argmax(y_conv, 1), feed_dict={x: [test_x], keep_prob: 1.0})
            test_value = test_y.argmax(0)
            test_value = 0 if flag == test_value else 1
            if test_value != answer[0]:
                print(test_value, answer[0])
                # print(test_value, answer[0])


def main():
    data = input_data.read_data_sets('MNIST_data', one_hot=True)

    sess = tf.InteractiveSession()

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    run_to_cal_accuracy(data, accuracy, train_step)
    run_to_estimate(data, accuracy, train_step)

if __name__ == '__main__':
    main()
