#!usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf

'''
conv1 ==> pool1 ==> conv2 ==> pool2 ==> dense ==> output
'''

batch_size = 32  # 每批训练大小
save_iter = 10  # 每训练save_iter批数据保存tensorboard结果
total_epochs = 5  # 总迭代轮次
learning_rate = 0.001
fold_time = 4  # 拼接词的个数（包括自身）

# Convolutional Layer #1
conv1_seq_length = 64  # img height, 假设句子中最多含有64个词，多余的丢弃
conv1_embedding_dim = 100  # img width,测试数据[100, 50, 25,5]
conv1_channels = 1  # img channels
conv1_kernel_size = (5, 5)
conv1_filters = 64

# Pooling Layer #1
pool1_size = (2, 2)
pool1_strides = 2

# Convolutional Layer #2
conv2_kernel_size = (5, 5)
conv2_filters = 32

# Pooling Layer #2
pool2_size = (2, 2)
pool2_strides = 2

# Dense Layer
dense_units_number = 1024
dropout_rate = 0.4

# Logits Layer
output_classes = 2


class TextCNNModel(object):
    def __init__(self):

        self.input_x = tf.placeholder(dtype=tf.float32,
                                      shape=(None, conv1_seq_length, conv1_embedding_dim),
                                      name="input_x")
        self.input_y = tf.placeholder(dtype=tf.float32,
                                      shape=(None, output_classes),
                                      name="input_y")
        self.mode = tf.placeholder(dtype=tf.string, name="mode")

        x_reshaped = tf.reshape(self.input_x, shape=(-1, conv1_seq_length, conv1_embedding_dim, conv1_channels))
        conv1 = tf.layers.conv2d(inputs=x_reshaped,
                                 filters=conv1_filters,
                                 kernel_size=conv1_kernel_size,
                                 activation=tf.nn.relu,
                                 name="conv1",
                                 padding="same")
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=pool1_size,
                                        strides=pool1_strides,
                                        name="pool1")
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=conv2_filters,
                                 kernel_size=conv2_kernel_size,
                                 activation=tf.nn.relu,
                                 name="conv2",
                                 padding="same")
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=pool2_size,
                                        strides=pool2_strides,
                                        name="pool2")

        pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
        dense = tf.layers.dense(inputs=pool2_flat,
                                units=dense_units_number,
                                activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense,
                                    rate=dropout_rate,
                                    training=self.mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout, units=output_classes)

        self.predict = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="y_pred_proba")
        }

        xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits)
        self.loss = tf.reduce_mean(xentropy)
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        # self.correct_pred = tf.equal(tf.argmax(input=self.input_y, axis=1), self.predict["classes"])
        self.input_y_idx = tf.argmax(input=self.input_y, axis=1)
        _, self.accuracy = tf.metrics.accuracy(labels=self.input_y_idx, predictions=self.predict["classes"])
        _, self.recall = tf.metrics.recall(labels=self.input_y_idx, predictions=self.predict["classes"])
        _, self.precision = tf.metrics.precision(labels=self.input_y_idx, predictions=self.predict["classes"])
        _, self.f1_score = tf.contrib.metrics.f1_score(labels=self.input_y_idx, predictions=self.predict["classes"])
