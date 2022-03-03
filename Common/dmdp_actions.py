import numpy as np
from enum import Enum

DIRECTION = [
    np.array([-1, 0]),
    np.array([1, 0]),
    np.array([0, -1]),
    np.array([0, 1]),
    np.array([0, 0]),
]

class actions(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
    stop = 4












