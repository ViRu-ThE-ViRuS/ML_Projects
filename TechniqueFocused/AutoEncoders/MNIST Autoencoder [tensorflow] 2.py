import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../Data/tmp/mnist/', one_hot=True)

LR = 0.05
numSteps = 30000
batchSize = 256
displaySteps = 1000
examplesToShow = 10
imgSize = 28

numH1 = 256
numH2 = 128
numInputs = 28 * 28

X = tf.placeholder(tf.float32, [None, numInputs])

weights = {
    'eH1': tf.Variable(tf.random_normal([numInputs, numH1])),
    'eH2': tf.Variable(tf.random_normal([numH1, numH2])),
    'dH1': tf.Variable(tf.random_normal([numH2, numH1])),
    'dH2': tf.Variable(tf.random_normal([numH1, numInputs]))
}

biases = {
    'eH1': tf.Variable(tf.zeros([numH1])),
    'eH2': tf.Variable(tf.zeros([numH2])),
    'dH1': tf.Variable(tf.zeros([numH1])),
    'dH2': tf.Variable(tf.zeros([numInputs]))
}


def encoder(x):
    en = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['eH1']), biases['eH1']))
    en = tf.nn.sigmoid(tf.add(tf.matmul(en, weights['eH2']), biases['eH2']))
    return en


def decoder(x):
    de = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['dH1']), biases['dH1']))
    de = tf.nn.sigmoid(tf.add(tf.matmul(de, weights['dH2']), biases['dH2']))
    return de


encoder = encoder(X)
decoder = decoder(encoder)

yPred = decoder
yTrue = X

loss = tf.reduce_mean(tf.pow(yTrue - yPred, 2))
optimizer = tf.train.RMSPropOptimizer(LR).minimize(loss)

print('Number of trainable parameters: {}'.format(
    np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1, numSteps + 1):
        batchX, _ = mnist.train.next_batch(batchSize)
        _, l = sess.run([optimizer, loss], feed_dict={X: batchX})

        if i % displaySteps == 0 or i == 1:
            print('Step {} Loss: {}'.format(i, l))

    n = 4
    canvasOrig = np.empty((28 * n, 28 * n))
    canvasRecon = np.empty((28 * n, 28 * n))
    for i in range(n):
        batchX, _ = mnist.test.next_batch(n)
        g = sess.run(decoder, feed_dict={X: batchX})

        for j in range(n):
            canvasOrig[i * 28:(i + 1) * 28, j * 28:(j + 1)
                                                   * 28] = batchX[j].reshape([28, 28])

        for j in range(n):
            canvasRecon[i * 28:(i + 1) * 28, j * 28:(j + 1)
                                                    * 28] = g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvasOrig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvasRecon, origin="upper", cmap="gray")
    plt.show()
