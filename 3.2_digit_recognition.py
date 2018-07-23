import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


print("loading data ...")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# optimize
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss=cross_entropy)

# 全局参数初始化器
tf.global_variables_initializer().run()

# 迭代地执行train_step， 每次feed数据给placeholder
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# prediction and accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # bool
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# test
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
