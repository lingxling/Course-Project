# coding=utf-8
"""
基本思路：用(2,2)的窗口最大池化原图片，再用HOG特征子得到图片特征向量。最后用SVM-线性核拟合
"""

import utils
import os
import sys
import cv2
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
mode = sys.argv[1]
father_path = os.path.join(os.getcwd(), "..")
hog_train_imgs_file = 'Q1/hog_train.npy'
hog_test_imgs_file = 'Q1/hog_test.npy'
pred_results_file = 'Q1/prediction.txt'
test_imgs_path = os.path.join(father_path, '测试数据及要求/test data')

if not os.path.exists('Q1'):
    os.mkdir('Q1')

if mode == 'predict' and os.path.exists(utils.train_imgs_file) and os.path.exists(utils.train_labels_file) \
        and os.path.exists(utils.test_imgs_file) and os.path.exists(utils.test_labels_file) \
        and os.path.exists(hog_train_imgs_file) and os.path.exists(hog_test_imgs_file):
    print('Predict BEGIN')
    _, Y_train, _, _ = utils.load_data()
    X_train = np.load(hog_train_imgs_file)
    X_test = np.load(hog_test_imgs_file)
elif mode == 'train' or not os.path.exists(utils.train_imgs_file) or not os.path.exists(utils.train_labels_file) \
        or not os.path.exists(utils.test_imgs_file) or not os.path.exists(utils.test_labels_file)\
        or not os.path.exists(hog_train_imgs_file) or not os.path.exists(hog_test_imgs_file):
    print('Train BEGIN')
    utils.repartition_data(father_path=os.path.abspath(os.path.join(os.getcwd(), "..")), test_rate=0)
    imgs_train, Y_train, _, _ = utils.load_data()
    X_train = utils.get_hog_features(imgs_train)
    # X_val = utils.get_hog_features(imgs_val)
    imgs_test = utils.get_imgs([test_imgs_path], max_pool=True, homomorphic=False,
                               file_type_list=['.bmp', '.png'], equalize=False, morphology=False)
    X_test = utils.get_hog_features(imgs_test)  # 获得HOG特征向量
    np.save(hog_train_imgs_file, X_train)
    np.save(hog_test_imgs_file, X_test)
else:
    raise Exception('Mode not defined')


if not os.path.exists('Q1/linear_svc.model') or mode == 'repartition':
    linear_svc = svm.LinearSVC(loss='hinge', max_iter=1000)
    linear_svc.fit(X_train, Y_train)
    pickle.dump(linear_svc, open('Q1/linear_svc.model', 'wb'))
linear_svc = pickle.load(open('Q1/linear_svc.model', 'rb'))
pred_test = linear_svc.predict(X_test)
files_name = os.listdir(test_imgs_path)
with open(pred_results_file, "w") as f:
    for i in range(len(files_name)):
        if pred_test[i] == 1:
            f.write(str(files_name[i]) + "    1\n")
        else:
            f.write(str(files_name[i]) + "    0\n")
# pred_val = linear_svc.predict(X_val)
# print('sklearn LinearSVC: {}'.format(accuracy_score(Y_val, linear_svc.predict(X_val))))


