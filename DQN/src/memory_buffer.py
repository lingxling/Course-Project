import random
import numpy as np
from collections import deque


class MemoryBuffer(object):
    def __init__(self, memory_size):
        self.buffer = deque()
        self.MAX_SIZE = memory_size

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) == self.MAX_SIZE:  # buffer is full
            self.buffer.popleft()
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))

    def size(self):
        return len(self.buffer)
