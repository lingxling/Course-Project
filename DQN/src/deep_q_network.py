from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
import numpy as np
import os


class DeepQNetwork(object):
    def __init__(self, input_shape, num_actions, name, learning_rate):
        self.input_shape = input_shape
        self.model = Sequential(name=name)
        self.model.add(Conv2D(32, 8, strides=(4, 4), activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(64, 4, strides=(2, 2), activation='relu'))
        self.model.add(Conv2D(64, 3, strides=(1, 1), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(num_actions))
        self.model.summary()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

    def plot_model(self, path, name):
        plot_model(self.model, to_file=os.path.join(path, name + '.png'))

    def copy_from(self, other):
        self.model.set_weights(other.model.get_weights())

    def train(self, states, y):
        self.model.train_on_batch(states, y)

    def get_action(self, state):
        pred = self.predict(np.array([state]))
        return np.argmax(pred[0])

    def predict(self, states):
        return self.model.predict(states)

    def save(self, path, name):
        self.model.save(os.path.join(path, name))

    def load(self, path, name):
        self.model = load_model(os.path.join(path, name))
