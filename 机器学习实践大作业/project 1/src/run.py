#!usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import time
from gensim.models import KeyedVectors
import utils
import cnn
import warnings
import sys


warnings.filterwarnings("ignore")  # 忽略警告信息
tensorboard_dir = "tensorboard"

save_dir = 'checkpoints/'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def select_hyperparameters_batch_size(wv):
    batch_size_list = [128, 64, 32, 16, 8, 4]
    for cur_batch_size in batch_size_list:
        cnn.batch_size = cur_batch_size
        total_accuracy, total_precision, total_recall, total_f1 = 0, 0, 0, 0
        start_time = time.time()
        for t in range(10):
            val_acc, val_pre, val_recall, val_f1 = train_and_test(wv)
            print(val_acc, val_pre, val_recall, val_f1)
            total_accuracy += val_acc
            total_precision += val_pre
            total_recall += val_recall
            total_f1 += val_f1
        print('Current Batch Size:', cur_batch_size, 'Time Used:', (time.time()-start_time)/10)
        print('Accuracy:', total_accuracy/10, 'Precision:', total_precision/10,
              'Recall:', total_recall/10, 'F1 Score:', total_f1/10)


def configure_saver():
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", cnn_model.loss)
    tf.summary.scalar("accuracy", cnn_model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return merged_summary, writer, saver


def validate(x_val, y_val, session):
    tmp_feed = {cnn_model.input_x: x_val,
                cnn_model.input_y: y_val,
                cnn_model.mode: tf.estimator.ModeKeys.EVAL}
    val_acc, val_pre, val_recall, val_f1 = session.run([cnn_model.accuracy, cnn_model.precision,
                                                        cnn_model.recall, cnn_model.f1_score],
                                                       feed_dict=tmp_feed)
    return val_acc, val_pre, val_recall, val_f1


def train_and_test(wv):
    # merged_summary, writer, saver = configure_saver()

    x_train, y_train = utils.get_processed_data(wv, train_file)
    # x_val, y_val = utils.get_processed_data(wv, val_file)
    x_test, y_test = utils.get_processed_data(wv, test_file)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    # writer.add_graph(session.graph)

    cur_batch = 0
    for x_batch, y_batch in utils.generate_batch(x_train, y_train):
        feed = {cnn_model.input_x: x_batch,
                cnn_model.input_y: y_batch,
                cnn_model.mode: tf.estimator.ModeKeys.TRAIN}
        session.run(cnn_model.train_op, feed_dict=feed)
        cur_batch += 1
    # val_acc, val_pre, val_recall, val_f1 = validate(x_val, y_val, session)
    test_acc, test_pre, test_recall, teste_f1 = validate(x_test, y_test, session)
    # saver.save(sess=session, save_path=save_path)
    return test_acc, test_pre, test_recall, teste_f1
    # return val_acc, val_pre, val_recall, val_f1


if __name__ == '__main__':
    mode = sys.argv[1]
    base_dir = sys.argv[2]
    train_file = os.path.join(base_dir, sys.argv[3])
    test_file = os.path.join(base_dir, sys.argv[4])
    contents_file = os.path.join(base_dir, 'contents')
    word2vec_wv_file = os.path.join(base_dir, "word2vec.wv")
    cnn.fold_time = 1
    cons, _ = utils.read_file(train_file)
    utils.write_contents(contents_file, cons)
    utils.write_word2vec_matrix(contents_file, word2vec_wv_file)
    word2vec_wv = KeyedVectors.load(word2vec_wv_file)
    cnn_model = cnn.TextCNNModel()
    t_accuracy, t_precision, t_recall, t_f1 = 0, 0, 0, 0
    t = time.time()
    for i in range(10):
        acc, pre, recall, f1 = train_and_test(word2vec_wv)
        print('iter', str(i)+':', acc, pre, recall, f1)
        t_accuracy += acc
        t_precision += pre
        t_recall += recall
        t_f1 += f1
    print("avg time used", (time.time() - t) / 10)
    print('Accuracy:', t_accuracy / 10, 'Precision:', t_precision / 10, 'Recall:', t_recall / 10, 'F1 Score:', t_f1 / 10)
