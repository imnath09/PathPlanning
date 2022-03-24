
import numpy as np

class InitMap():
    def __init__(self):
        self.height = 30
        self.width = 16
        self.start = np.array([0, 3])
        self.destination = np.array([29, 12])
        #self.destination = np.array([12, 7])

        self.obstacles = []

        for i in range(8):
            #for j in range(1):
            self.obstacles.append(np.array([4 + 3 * i, 2]))# + 6 * j]))
            self.obstacles.append(np.array([4 + 3 * i, 3]))# + 6 * j]))
            self.obstacles.append(np.array([4 + 3 * i, 4]))# + 6 * j]))
            self.obstacles.append(np.array([4 + 3 * i, 5]))# + 6 * j]))

        for i in range(6):
            #for j in range(2):
            self.obstacles.append(np.array([4 + 4 * i, 11]))# + 5 * j]))
            self.obstacles.append(np.array([4 + 4 * i, 12]))# + 5 * j]))
            self.obstacles.append(np.array([4 + 4 * i, 13]))# + 5 * j]))
            self.obstacles.append(np.array([5 + 4 * i, 11]))# + 5 * j]))
            self.obstacles.append(np.array([5 + 4 * i, 12]))# + 5 * j]))
            self.obstacles.append(np.array([5 + 4 * i, 13]))# + 5 * j]))

        for j in range(3):
            self.obstacles.append(np.array([4 + 8 * j, 8]))
            self.obstacles.append(np.array([5 + 8 * j, 8]))
            self.obstacles.append(np.array([6 + 8 * j, 8]))
            self.obstacles.append(np.array([7 + 8 * j, 8]))
            self.obstacles.append(np.array([8 + 8 * j, 8]))
            self.obstacles.append(np.array([9 + 8 * j, 8]))
