# -*- coding: UTF-8 -*-

import utils
import os
import copy
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from collections import Counter

if not os.path.exists('Q1'):
    os.mkdir('Q1')
if not os.path.exists('Q1/data_0'):
    os.mkdir('Q1/data_0')
if not os.path.exists('Q1/test_data_0'):
    os.mkdir('Q1/test_data_0')
father_dir_path = os.path.abspath(os.path.join(os.path.join(os.getcwd(), "..")))
read_dir_path = os.path.join(father_dir_path, '数据')
test_dir_path = os.path.join(father_dir_path, 'test data')

useless_infos = ['姓名', '患者名', '职业', '入院时间', '婚姻状况', '患者签名', '陈述者签名', '与患者关系', '入科时间',
                 '记录时间', '病史陈述者', '民族', '患者名\n', '病史记录已阅，与陈述相符，属实。\n', '出生地', '发病日期',
                 '发病节气', '入院记录\n', '影像号']  # 在原文本中，包含这些内容的行非常混乱，直接去掉
to_delete_files = ['肾病_2012_192522.txt', '肾病_2014_163726.txt', '肾病_2014_102213.txt', '肾病_2012_238004.txt',
                   '肾病_2012_331060.txt', '酮症_2011_264972.txt', '肾病_2013_361889.txt', '肾病_2012_60355.txt',
                   '肾病_2014_193373.txt', '心脏病_84902.txt', '心脏病_301132.txt', '足病_299975.txt',
                   '酮症_2009_245198.txt', '酮症_2011_119535.txt']  # 这些样本没有包含有用的信息，直接删除

label_to_disease = {0: '肾病', 1: '酮症', 2: '心脏病', 3: '眼病', 4: '周围神经病', 5: '足病'}

file_infos, file_names = utils.read_data(read_dir_path, to_delete_files, useless_infos)
utils.write_data(file_infos, file_names, 'Q1/data_0')
utils.write_segmented_file(file_infos, 'Q1/segmented_line.txt')
utils.write_format_csv_file(padding=False)

test_file_infos, test_file_names = utils.read_data(test_dir_path, useless_infos=useless_infos)
utils.write_data(test_file_infos, test_file_names, 'Q1/test_data_0')
utils.write_segmented_file(test_file_infos, 'Q1/test_segmented_line.txt')
utils.write_format_csv_file(dir_path='Q1/test_data_0', format_info_save_file='Q1/test_format_info.csv', padding=False)
test_words, test_word_weight_matrix = utils.get_tfidf_words_and_weight_matrix('Q1/test_segmented_line.txt')

words, word_weight_matrix = utils.get_tfidf_words_and_weight_matrix('Q1/segmented_line.txt')
labels = utils.get_Q1_labels(dir_path='Q1/data_0')
legal_words = set(set(words).intersection(set(test_words)))
X = utils.get_X(words, word_weight_matrix, legal_words=legal_words)
test_samples = utils.get_X(test_words, test_word_weight_matrix, legal_words=legal_words)

X_train, _, Y_train, _ = train_test_split(X, labels, shuffle=True, test_size=0)

Y_train_1 = copy.deepcopy(Y_train)

Y_train_1[Y_train_1 == 0.0] = -1
Y_train_1[Y_train_1 == 1.0] = -1
Y_train_1[Y_train_1 == 4.0] = -1
Y_train_1[Y_train_1 != -1] = 1

sub_X_train_1, sub_Y_train_1 = utils.get_samples_with_specific_label(X_train, Y_train, specific_labels=[0.0, 1.0, 4.0])
sub_X_train_2, sub_Y_train_2 = utils.get_samples_with_specific_label(X_train, Y_train, specific_labels=[2.0, 3.0, 5.0])

# 训练第1个分类器，将数据集分为{0, 1, 4}和{2, 3}两类
clf_1 = svm.LinearSVC()
clf_1.fit(X_train, Y_train_1)

# 训练第2个分类器，将0, 1, 4分开
clf_2 = svm.LinearSVC()
clf_2.fit(sub_X_train_1, sub_Y_train_1)

# 训练第3个分类器，将2，3，5分开
clf_3 = svm.LinearSVC()
clf_3.fit(sub_X_train_2, sub_Y_train_2)

# 预测

for i in ['1', '2', '3', '4', '5', '6']:
    if not os.path.exists('Q1/'+i):
        os.mkdir('Q1/'+i)

pred = np.zeros(test_samples.shape[0])

for i in range(test_samples.shape[0]):
    cur_sample = [test_samples[i]]
    y = clf_1.predict(cur_sample)[0]
    if y == -1.0:
        cur_pred = clf_2.predict(cur_sample)[0]
    else:
        cur_pred = clf_3.predict(cur_sample)[0]

    cur_file_target_dir = os.path.join('Q1', str(int(cur_pred+1)))
    shutil.copyfile(os.path.join(test_dir_path, test_file_names[i]), os.path.join(cur_file_target_dir, test_file_names[i]))
