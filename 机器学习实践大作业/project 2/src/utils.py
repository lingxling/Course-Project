import os
import numpy as np
import cv2

from skimage.feature import hog
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split

# 经过池化处理的完整数据集
train_imgs_file = 'Q1/max_pooling_train_imgs.npy'
train_labels_file = 'Q1/max_pooling_train_labels.npy'
test_imgs_file = 'Q1/max_pooling_test_imgs.npy'
test_labels_file = 'Q1/max_pooling_test_labels.npy'


def adjust_gamma(image, gamma=1.0, c=1.0):
    tmp = c*np.power(image/255, gamma)*255
    return tmp.astype(np.uint8)


def laplacian_sharpen(img):
    img = cv2.GaussianBlur(img, (3, 3), 3)
    img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    return cv2.convertScaleAbs(img)


class HomomorphicFilter(object):
    def __init__(self, gammaL=0.75, gammaH=1.25, filter_parameters=(10, 2)):
        """
        :param gammaL: (gammaH-gammaL)
        :param gammaH: (gammaH-gammaL)
        :param filter_parameters: filter_parameters[0]: D0, filter size, larger D0 correspond to
        """
        self.gammaL = float(gammaL)
        self.gammaH = float(gammaH)
        self.filter_params = filter_parameters

    def __butterworth_filter(self, fft_img_shape):
        P = fft_img_shape[0]/2
        Q = fft_img_shape[1]/2
        #中心化操作
        U, V = np.meshgrid(range(fft_img_shape[0]), range(fft_img_shape[1]), sparse=False, indexing='ij')
        Duv = ((U-P)**2 + (V-Q)**2).astype(float)
        Huv = 1/(1+(Duv/self.filter_params[0])**(2*self.filter_params[1]))
        return 1 - Huv

    def __gaussian_filter(self, fft_img_shape, c=0.2):
        P = fft_img_shape[0]/2
        Q = fft_img_shape[1]/2
        # 中心化操作
        U, V = np.meshgrid(range(fft_img_shape[0]), range(fft_img_shape[1]), sparse=False, indexing='ij')
        Duv = ((U-P)**2 + (V-Q)**2).astype(float)
        Huv = np.exp((-c*Duv**2/(2*(self.filter_params[0])**2)))
        return 1 - Huv

    def __apply_filter(self, fft_img, H):
        H = np.fft.fftshift(H)
        H = self.gammaL + (self.gammaH-self.gammaL)*H
        return H*fft_img

    def filter(self, img, filter_name='gaussian'):
        """
        :param img: single channel image
        :param filter_name: options: gaussian, butterworth
        :return: filtered image
        """
        log_img = np.log1p(np.array(img, dtype=float))
        fft_img = np.fft.fft2(log_img)
        if filter_name == 'gaussian':
            Huv = self.__gaussian_filter(fft_img.shape)
        elif filter_name == 'butterworth':
            Huv = self.__butterworth_filter(fft_img.shape)
        else:
            raise Exception('Selected filter not implemented')

        filtered_fft_img = self.__apply_filter(fft_img, Huv)
        inverse_filtered_img = np.fft.ifft2(filtered_fft_img)
        img = np.exp(np.real(inverse_filtered_img)) - 1
        return np.uint8(img)


def cv2_imread(filepath):
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def get_paths(father_path=os.path.abspath(os.path.join(os.getcwd(), ".."))):
    """
    :param father_path: the path where '作业数据集' exists
    :return: 2 path list of directory where images exist and no exists
    """
    data_path = os.path.join(father_path, '作业数据集')
    hand_path_list = []
    no_hand_path_list = []
    for data_dir in os.listdir(data_path):
        if data_dir[0] == '不':
            no_hand_path_list.append(os.path.join(data_path, data_dir))
        else:
            hand_path_list.append(os.path.join(data_path, data_dir))
    return hand_path_list, no_hand_path_list


def max_pooling(imgs):
    # img = np.asarray(img, dtype='float32')/256
    max_pool_imgs = np.zeros(shape=(imgs.shape[0], imgs.shape[1]//2, imgs.shape[2]//2))
    for i in range(imgs.shape[0]):
        max_pool_imgs[i] = block_reduce(imgs[i], (2, 2), np.max)
    del imgs
    return max_pool_imgs


def get_img_number(dirpath_list, img_type_list):
    files_num = 0
    for dirpath in dirpath_list:
        files = os.listdir(dirpath)
        for file in files:
            if file[-4:] in img_type_list:
                files_num += 1
    return files_num


def get_imgs(dirpath_list, max_pool=True, homomorphic=False, file_type_list=['.bmp', '.png'],
             equalize=False, morphology=False, bit_wise=False):
    """
    :param dirpath_list: list, contains directory paths
    :param max_pool: bool, max_pooling or not
    :param homomorphic: bool, do homomorphic or not
    :param file_type: str [len(str)=4], the file type which we are going to read
    :return:
    """
    # 获取所有图片的路径，并排序
    img_path_list = []
    for dirpath in dirpath_list:
        files = os.listdir(dirpath)
        for file in files:
            if file[-4:] in file_type_list:
                img_path_list.append(os.path.join(dirpath, file))
    img_path_list.sort()
    # print(img_path_list)
    test_img = cv2_imread(img_path_list[0])
    h = test_img.shape[0]
    w = test_img.shape[1]
    # print(h, w)
    files_num = get_img_number(dirpath_list, file_type_list)
    homo_filter = HomomorphicFilter()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if max_pool:
        imgs = np.zeros(shape=(files_num, h//2, w//2), dtype=np.float32)
    else:
        imgs = np.zeros(shape=(files_num, h, w), dtype=np.float32)
    cnt = 0
    for img_path in img_path_list:
        img = cv2_imread(img_path)
        # cv2.imshow('init', img)

        if homomorphic:
            img = homo_filter.filter(img, filter_name='gaussian')
        if equalize:
            img = cv2.equalizeHist(img)
        if morphology:
            dilated = cv2.dilate(img, kernel)
            eroded = cv2.erode(img, kernel)
            img = cv2.absdiff(dilated, eroded)
        if bit_wise:
            img = cv2.bitwise_not(img)
            # _, img = cv2.threshold(edge, 5, 255, cv2.THRESH_BINARY)
            # img = laplacian_sharpen(img)
        if max_pool:
            img = block_reduce(img, (2, 2), np.max)
        # cv2.imshow('after', img)
        # cv2.waitKey(0)
        imgs[cnt] = img
        cnt += 1
    return imgs


def get_mask_img(dirpath_list, max_pool=True, file_type_list=['.bmp', '.png']):
    """
    :param max_pool: bool, max_pooling image or not
    :param file_type: str, the file type of label image
    :return:
    """
    # 获取所有图片的路径，并排序
    img_path_list = []
    for dirpath in dirpath_list:
        files = os.listdir(dirpath)
        for file in files:
            if file[-4:] in file_type_list:
                img_path_list.append(os.path.join(dirpath, file))
    img_path_list.sort()
    test_img = cv2_imread(img_path_list[0])
    h = test_img.shape[0]
    w = test_img.shape[1]
    label_img_number = get_img_number(dirpath_list, file_type_list)
    if max_pool:
        imgs = np.zeros(shape=(label_img_number, h//2, w//2), dtype=np.float32)
    else:
        imgs = np.zeros(shape=(label_img_number, h, w), dtype=np.float32)
    cnt = 0
    for img_path in img_path_list:
        img = cv2_imread(img_path)
        if max_pool:
            img = block_reduce(img, (2, 2), np.max)
        img[img > 0] = 1
        img[img == 0] = 0
        imgs[cnt] = img
        cnt += 1

    return imgs


def get_train_and_test_data(X, Y, train_rate=0.8):
    indices = np.random.permutation(np.arange(len(Y)))
    boundary = int(len(Y) * train_rate)
    X_shuffle = X[indices]
    Y_shuffle = Y[indices]
    del X, Y
    return X_shuffle[:boundary], Y_shuffle[:boundary], X_shuffle[boundary:], Y_shuffle[boundary:]


def save_files(imgs_train, Y_train, imgs_test, Y_test):
    np.save(train_imgs_file, imgs_train)
    np.save(train_labels_file, Y_train)
    np.save(test_imgs_file, imgs_test)
    np.save(test_labels_file, Y_test)
    print('SAVE SUCCESSFUL!')


def get_hog_features(imgs):
    imgs_hog_features = np.zeros(shape=(imgs.shape[0], (29*39*2*2*8)))
    cnt = 0
    for img in imgs:
        # print(img.shape)
        fd = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 visualize=False, multichannel=False, feature_vector=True)
        # print(fd.shape)
        imgs_hog_features[cnt] = fd
        cnt += 1
    print('GET HOG FEATURES!')
    return imgs_hog_features


def repartition_data(father_path=os.path.abspath(os.path.join(os.getcwd(), "..")), test_rate=0.25):
    hand_paths, no_hand_paths = get_paths(father_path=father_path)
    positive_imgs = get_imgs(hand_paths)
    negative_imgs = get_imgs(no_hand_paths)
    positive_labels = np.ones(shape=(positive_imgs.shape[0]), dtype=bool)
    negative_labels = np.zeros(shape=(negative_imgs.shape[0]), dtype=bool) - 1
    labels = np.concatenate((positive_labels, negative_labels), axis=0)
    imgs = np.concatenate((positive_imgs, negative_imgs), axis=0)
    del positive_imgs, positive_labels, negative_imgs, negative_labels
    # X_train, Y_train, X_test, Y_test = get_train_and_test_data(imgs, labels)

    imgs_train, imgs_test, Y_train, Y_test = train_test_split(imgs, labels, shuffle=True, test_size=test_rate)
    # print(imgs_train.shape, imgs_test.shape, Y_train.shape, Y_test.shape)
    del imgs, labels
    save_files(imgs_train, Y_train, imgs_test, Y_test)
    del imgs_train, Y_train, imgs_test, Y_test


def load_data():
    imgs_train = np.load(train_imgs_file)
    Y_train = np.load(train_labels_file)
    imgs_test = np.load(test_imgs_file)
    Y_test = np.load(test_labels_file)
    print('LOAD SUCCESSFUL!')
    return imgs_train, Y_train, imgs_test, Y_test
