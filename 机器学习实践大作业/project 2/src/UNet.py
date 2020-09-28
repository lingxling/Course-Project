import utils
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K


class UNet(object):
    def __init__(self, img_shape=(480, 640, 1)):
        self.img_shape = img_shape
        inputs = keras.layers.Input(shape=self.img_shape)
        conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = keras.layers.Dropout(0.5)(conv4)  # 加快计算速度

        pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
        conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = keras.layers.Dropout(0.5)(conv5)

        up6 = keras.layers.Conv2D(512, 2, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(drop5))
        merge6 = keras.layers.concatenate([conv4, up6], axis=3)
        conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = keras.layers.Conv2D(256, 2, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = keras.layers.Conv2D(128, 2, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = keras.layers.Conv2D(64, 2, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        self.model = keras.models.Model(inputs=inputs, outputs=conv10)

    @staticmethod
    def dice(y_true, y_pred, smooth=1):
        intersection_sum = tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return (2. * intersection_sum + smooth) / (denominator + smooth)

    def dice_loss(self, y_true, y_pred):
        return (1 - self.dice(y_true, y_pred))*100

    @staticmethod
    def iou(y_true, y_pred, smooth=1):
        intersection_sum = tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        union = denominator - intersection_sum
        return (intersection_sum + smooth) / (union + smooth)

    def iou_loss(self, y_true, y_pred):
        return (1 - self.iou(y_true, y_pred))*100

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return (1 - self.dice_coef(y_true, y_pred))*100

    @staticmethod
    def jaccard_distance_loss(y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * 100

    def __unet_model(self):
        self.model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss=keras.losses.binary_crossentropy,
                           metrics=[self.iou_loss, self.dice_loss])

    def save_unet(self, path='Q2/'):
        self.model.save_weights(path + 'model_weights')  # save the weights
        with open(path + 'model_architecture.json', 'w') as f:  # save the model architecture
            f.write(self.model.to_json())

    def load_unet(self,  path='Q2/'):
        with open(path + 'model_architecture.json', 'r') as f:  # model reconstruction from JSON file
            self.model = keras.models.model_from_json(f.read())
        self.model.load_weights(path + 'model_weights')  # load weights into the self.model

    def train(self, imgs, labels, batch_size=32):
        self.__unet_model()
        earlystopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=5,
                                                      restore_best_weights=True)
        self.model.fit(imgs, labels, batch_size=batch_size, epochs=10, shuffle=True,
                       validation_split=0, callbacks=[earlystopping])

    def predict(self, X_test, batch_size=32):
        return self.model.predict(X_test, batch_size=batch_size)

    def evaluate(self, X_test, Y_test, batch_size=32):
        self.__unet_model()
        return self.model.evaluate(X_test, Y_test, batch_size=batch_size)