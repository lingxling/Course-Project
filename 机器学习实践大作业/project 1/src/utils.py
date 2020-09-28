#!usr/bin/env python
# -*- coding:utf-8 -*-

import re
import numpy as np
import time
import cnn
from tensorflow import keras as kr
from gensim.models import word2vec


def get_time_dif(start_time):
    return time.time() - start_time


def read_file(filename):
    rule = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9《》\t ]")  # 只保留中英文，数字，空格和制表符
    contents, labels = [], []
    with open(filename, mode='r', encoding='utf-8') as fr:
        for line in fr:
            line_arr = rule.sub('', line).split()
            labels.append([int(line_arr[0])])
            contents.append(line_arr[1:])
    return contents, labels


def write_contents(filename, contents, split_sign=' '):
    with open(filename, mode='w', encoding='utf-8') as f:
        for words in contents:
            for word in words:
                print(word, end=split_sign, file=f)
            print(file=f)


def write_word2vec_matrix(contents_file, wv_target_file):
    print('Train word2vec Matrix...')
    begin = time.time()
    word2vec_model = word2vec.Word2Vec(word2vec.LineSentence(contents_file),
                                       size=cnn.conv1_embedding_dim//cnn.fold_time,
                                       window=5,
                                       min_count=1)
    word2vec_model.wv.save(wv_target_file)
    print("Time Usage:", get_time_dif(begin))
    del word2vec_model


def sentence_to_matrix(sentence, wv, seq_length=cnn.conv1_seq_length):
    mat_np = np.zeros(shape=(cnn.conv1_seq_length, cnn.conv1_embedding_dim))
    cnt = 0
    for i in range(len(sentence)):
        if sentence[i] in wv:
            mat_np[cnt] = wv[sentence[i]]
            cnt += 1
        if cnt == seq_length:
            return mat_np
    return mat_np


def get_sentences_matrix(wv, contents):
    res = np.zeros((len(contents), cnn.conv1_seq_length, cnn.conv1_embedding_dim))
    for i in range(len(contents)):
        res[i] = sentence_to_matrix(contents[i], wv)
    return res


def generate_batch(x, y, batch_size=cnn.batch_size):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def get_processed_data(wv, filename):
    contents, labels = read_file(filename)
    x_train = get_sentences_matrix(wv, contents)
    y_train = kr.utils.to_categorical(labels, num_classes=cnn.output_classes)
    return x_train, y_train

