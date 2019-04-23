import numpy as np

import gym

class GammaDist():
    def __init__(self, k, theta):
        self.k = np.array(k)
        self.theta = np.array(theta)
        self.size = self.k.size
        self.shape = self.k.shape

    def draw(self):
        return np.random.gamma(self.k, self.theta)

    def draw_row(self, i):
        return np.random.gamma(self.k[i], self.theta[i])

    def draw_elem(self, x):
        assert x >= 0
        i = x // self.shape[1]
        j = x % self.shape[1]
        return np.random.gamma(self.k[i][j], self.theta[i][j])

    def draw_i_j(self, i, j):
        return np.random.gamma(self.k[i][j], self.theta[i][j])

    def __repr__(self):
        return "(GammaDist: k = {k}, width = {theta})".format(
            k = self.k,
            theta = self.theta
        )

def row_distance(shape, a, b):
    # shape: the shape of the matrix on which we want to calculate row distance
    # a, b: coordinates to compare, row-major
    # everything is numpy arrays
    d_col = abs(a[1] - b[1])
    d_row = min(a[0] + b[0], 2 * shape[0] - a[0] - b[0])
    # Add 1 to all distances to avoid repeat sampling of same plot forever
    return abs(a[0] - b[0]) + 1 if a[1] == b[1] else sum((d_col, d_row)) + 1

def build_distance_matrix(shape, n_elements):
    D = np.zeros((n_elements, n_elements))
    for e_1 in range(n_elements):
        a = (e_1 // shape[1], e_1 % shape[1])
        for e_2 in range(n_elements):
            b = (e_2 // shape[1], e_2 % shape[1])
            d_e = row_distance(shape, a, b)
            D[e_1][e_2] = d_e
            D[e_2][e_1] = d_e
    return D

class FieldEnv(gym.Env):
    first_round = True
    def __init__(self):
        self.reinit()
        n_elements = self.k.size

        self.gamma = GammaDist(self.k, self.theta)
        self.D = build_distance_matrix(self.k.shape, n_elements)

        self.action_space = gym.spaces.Discrete(n_elements)
        self.observation_space = gym.spaces.Discrete(self.D.max() + 1)

        self.C = 500
        self.accs = np.array(())

        self.seed()
        self.reset()

    def reinit(self):
        self.k = np.linspace(1.25, 2.25, num = 100).reshape((10,10))
        np.random.shuffle(self.k)
        self.theta = np.tile(0.125, self.k.shape)

        self.position = 0
        self.observation = 0
        self.sigma_c = 0

        self.times_picked = np.zeros(100)
        self.mean_rewards = np.zeros(100)

        self.actions = np.zeros(1)

        return self.observation

    def step(self, action):
        assert self.action_space.contains(action)

        self.reward = self.gamma.draw_elem(action)

        # Distance traveled = observation
        self.distance_traveled = self.D[self.position][action]
        self.sigma_c += self.distance_traveled
        done = self.sigma_c >= self.C

        # position_last = self.position
        self.position = action
        self.actions = np.append(self.actions, action)
        #self.observation = np.array([(position_last, position, self.distance_traveled,
        #    self.reward - (self.times_picked[action] / (self.times_picked.sum() + 1)))])
        self.observation = self.distance_traveled
        self.mean_rewards[action] = ((self.reward / (self.times_picked[action] + 1)) +
            (self.mean_rewards[action] / (self.times_picked[action] + 1)) * self.times_picked[action])
        self.times_picked[action] += 1

        return self.observation, self.reward, done, {"position": self.position, "reward": self.reward}

    def reset(self):
        top_ten_ests = np.flip(np.argsort(self.mean_rewards))[0:10]
        top_ten_true = np.flip(np.argsort(self.k, None))[0:10]
        self.acc = np.mean(np.isin(top_ten_ests, top_ten_true))
        np.savetxt("top_10_accs.csv", np.array([self.acc]), delimiter = ",")
        np.savetxt("actions.csv", self.actions, delimiter = ",")
        return self.reinit()
