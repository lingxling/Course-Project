# coding=utf-8
"""
基本策略：先用max_pooling缩小图片，用UNET预测。然后将预测得到的图片做线性插值，恢复为原来的样子。
"""
import sys
import os
import numpy as np
import UNet
import utils
import cv2

mode = sys.argv[1]
PREDICT = 'predict'
TRAIN = 'train'
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 形态学处理（膨胀、腐蚀等）所用的核

# 获取原始训练图像和掩码图像
imgs = utils.get_imgs(['init_imgs'], file_type_list=['.bmp', '.png'], homomorphic=True,
                      max_pool=False, morphology=True, bit_wise=True)
mask = utils.get_imgs(['mask_imgs'], file_type_list=['.bmp', '.png'], max_pool=False,
                      morphology=False, bit_wise=True)

# 预处理用于训练的输入图像和掩码图像并保存
if not os.path.exists('Q2'):
    mode = TRAIN
    os.mkdir('Q2')
if not os.path.exists('Q2/imgs'):
    os.mkdir('Q2/imgs')
if not os.path.exists('Q2/mask_imgs'):
    os.mkdir('Q2/mask_imgs')
for i in range(imgs.shape[0]):
    tmp = imgs[i].astype(np.uint8)
    tmp = utils.adjust_gamma(tmp, gamma=6)
    tmp = cv2.erode(tmp, kernel)
    tmp = cv2.dilate(tmp, kernel)
    cv2.imwrite('Q2/imgs/' + str(i) + '.bmp', tmp)
for i in range(mask.shape[0]):
    tmp = mask[i].astype(np.uint8)
    # tmp [tmp > 0] = 255
    tmp[tmp < 255] = 0
    cv2.imwrite('Q2/mask_imgs/' + str(i) + '.bmp', tmp)

# 预处理用于测试的图像并保存
if not os.path.exists('Q2/predictions'):
    os.mkdir('Q2/predictions')
if not os.path.exists('Q2/test_imgs'):
    os.mkdir('Q2/test_imgs')
father_path = os.path.join(os.getcwd(), "..")
test_imgs_dir_path = os.path.join(father_path, '测试数据及要求/test data')
test_imgs = utils.get_imgs([test_imgs_dir_path], max_pool=False, homomorphic=True,
                           file_type_list=['.bmp', '.png'], morphology=True, bit_wise=True)
test_imgs_name_list = os.listdir(test_imgs_dir_path)
test_imgs_name_list.sort()
for i in range(test_imgs.shape[0]):
    tmp = test_imgs[i].astype(np.uint8)
    tmp = utils.adjust_gamma(tmp, gamma=6)
    tmp = cv2.erode(tmp, kernel)
    tmp = cv2.dilate(tmp, kernel)
    cv2.imwrite('Q2/test_imgs/' + test_imgs_name_list[i], tmp)

if mode == PREDICT:
    test_imgs_name_list = os.listdir('Q2/test_imgs')
    test_imgs_name_list.sort()
    imgs_test = utils.get_imgs(['Q2/test_imgs'], max_pool=True, file_type_list=['.bmp', '.png']) / 255
    test_shape = imgs_test.shape
    X_test = imgs_test.reshape(test_shape[0], test_shape[1], test_shape[2], 1)

    print('****************Predict BEGIN****************')
    unet = UNet.UNet(img_shape=(test_shape[1], test_shape[2], 1))
    unet.load_unet()
    Y_pred = unet.predict(X_test, batch_size=2)
    for i in range(Y_pred.shape[0]):
        cur_pred = Y_pred[i].reshape(test_shape[1], test_shape[2])
        cur_pred[cur_pred >= 0.5] = 255
        cur_pred[cur_pred < 0.5] = 0
        cur_pred = cv2.resize(cur_pred, (640, 480), interpolation=cv2.INTER_AREA)
        cur_pred = cur_pred.astype(np.uint8)
        cur_pred = cv2.bitwise_not(cur_pred)
        cur_name = test_imgs_name_list[i][:-4]
        cv2.imwrite('Q2/predictions/' + cur_name + '_mask.bmp', cur_pred)
    print('****************Predict END****************')

elif mode == TRAIN:
    imgs = utils.get_imgs(['Q2/imgs'], file_type_list=['.bmp'], max_pool=True) / 255
    mask = utils.get_mask_img(['Q2/mask_imgs'], file_type_list=['.bmp'], max_pool=True)
    train_shape = imgs.shape
    X_train = imgs.reshape(train_shape[0], train_shape[1], train_shape[2], 1)
    Y_train = mask.reshape(train_shape[0], train_shape[1], train_shape[2], 1)
    print('****************Train BEGIN****************')
    unet = UNet.UNet(img_shape=(train_shape[1], train_shape[2], 1))
    unet.load_unet()
    unet.train(X_train, Y_train, batch_size=2)
    unet.save_unet()
    print('****************Train END****************')

else:
    raise Exception('Mode not defined')


