import numpy as np
import math

EPS_MIN = 0.01
EPS_MAX = 1.0
EPS_DECAY = 10000


def epsilon_greedy(step):
    epsilon = EPS_MIN + (EPS_MAX - EPS_MIN) * math.exp(-1. * step / EPS_DECAY)
    return np.random.rand() < epsilon