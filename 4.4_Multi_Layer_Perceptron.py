from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()


"""
1. 算法设计，前馈网络
2. 损失函数和优化器
3. 批训练
4. 模型准确率评测
"""

in_units = 784
h1_units = 300
w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
hidden1_dropout = tf.nn.dropout(hidden1, keep_prob=keep_prob)

y = tf.nn.softmax(tf.matmul(hidden1_dropout, w2) + b2)


# 损失函数，优化器
y_ = tf.placeholder(tf.float32, [None, 10])
# loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(learning_rate=0.3).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# 批训练
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    if i % 1000 == 0:
        # print(sess.run(loss))
        print(cross_entropy.run())

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 测试集
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

