# -*- coding: UTF-8 -*-

import os
import sys
import utils
import numpy as np
import copy
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm, naive_bayes, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE  # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
import pickle

if not os.path.exists('Q2'):
    os.mkdir('Q2')
if not os.path.exists('Q2/data_0'):
    os.mkdir('Q2/data_0')
father_dir_path = os.path.abspath(os.path.join(os.path.join(os.getcwd(), "..")))
read_dir_path = os.path.join(father_dir_path, '数据')
useless_infos = ['姓名', '患者名', '职业', '入院时间', '婚姻状况', '患者签名', '陈述者签名', '与患者关系', '入科时间',
                 '记录时间', '病史陈述者', '民族', '患者名\n', '病史记录已阅，与陈述相符，属实。\n', '出生地', '发病日期',
                 '发病节气', '入院记录\n', '影像号']  # 在原文本中，包含这些内容的行非常混乱，直接去掉
to_delete_files = ['肾病_2012_192522.txt', '肾病_2014_163726.txt', '肾病_2014_102213.txt', '肾病_2012_238004.txt',
                   '肾病_2012_331060.txt', '酮症_2011_264972.txt', '肾病_2013_361889.txt', '肾病_2012_60355.txt',
                   '肾病_2014_193373.txt', '心脏病_84902.txt', '心脏病_301132.txt', '足病_299975.txt',
                   '酮症_2009_245198.txt', '酮症_2011_119535.txt']  # 这些样本没有包含有用的信息，直接删除

label_to_disease = {0: '肾病', 1: '酮症', 2: '心脏病', 3: '眼病', 4: '周围神经病', 5: '足病'}

file_infos, file_names = utils.read_data(read_dir_path, to_delete_files, useless_infos, is_Q1=False)  # is_Q1指明要过滤和保存某些信息
utils.write_data(file_infos, file_names, 'Q2/data_0')
utils.write_segmented_file(file_infos, 'Q2/segmented_line.txt')
utils.write_format_csv_file(padding=False)

# 测试1：预测住院天数
labels = utils.get_Q2_labels(dir_path='Q2/data_0')
words, word_weight_matrix = utils.get_tfidf_words_and_weight_matrix('Q2/segmented_line.txt')
X = utils.get_X(words, word_weight_matrix, legal_words=words)
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, shuffle=True, test_size=0.2)
reg = svm.LinearSVR()
reg.fit(X_train, Y_train)
pred = reg.predict(X_test)
print('方差可解释性', metrics.explained_variance_score(Y_test, pred))
print('绝对误差', metrics.mean_absolute_error(Y_test, pred))
print('均方误差', metrics.mean_squared_error(Y_test, pred))
