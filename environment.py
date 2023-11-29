import osmnx as ox;
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import scipy.stats
from utils import *


class Environment:
    """
  Simulation of graph route that can be deterministic and stochastic.
  The main functionalities is to reset the simulation and returns the state based in the action.
  """

    def __init__(self, G, source, target, reward="weighted", mode="deterministic"):
        self.G = G
        self.max_weight = max([G[u][v][0]["length"] for u, v, m in G.edges if m == 0])
        self.source = source
        self.target = target
        self.reward = reward
        self.v_nodes = {n: 0 for n in list(self.G.nodes())}
        if mode == "deterministic":
            self.step = self.step_deterministic
        elif mode == "stochastic":
            self.step = self.step_stochastic

    def get_n_states(self):
        """Get number of states"""
        return len(self.G.nodes)

    def reset(self):
        """Return to the source state that is not the destination"""

        self.state = self.source
        # self.state = np.random.choice(list(self.G.nodes))
        # while self.state == self.target:
        # self.state = np.random.choice(list(self.G.nodes))
        return self.state

    def step_deterministic(self, action):
        """Return new state, reward, and if simulation is done"""

        # v_n = self.v_nodes.get(self.state)
        # self.v_nodes.update([(self.state, v_n + 1)])

        # if is not the target, and the action is to stay in the same node
        if (action == self.state) & (action != self.target):
            return self.state, -1, False

        # if reached the target
        if self.state == self.target:
            return self.state, 10000, True

        # weight of the edge
        w = self.G[self.state][action][0]["length"] / self.max_weight
        # now, the state is the next node
        self.state = action

        # if the action leaves to a dead end
        neighbors = list(self.G.neighbors(self.state))
        if len(neighbors) == 0:
            return self.state, -1, True

        if self.reward == "unit":
            return self.state, 0, False
        elif self.reward == "weighted":
            return self.state, -w, False

    def step_stochastic(self, action):
        """Return new state, reward, and if the simulation is done"""

        if np.random.rand() < 0.05:  # 5% probability of not moving
            return self.state, 0, False

        if self.state == self.target:
            return self.state, 10000, True

        if (action == self.state) & (action != self.target):
            return self.state, -1, False

        w = self.G[self.state][action][0]["length"] / self.max_weight

        self.state = action

        # if the action leaves to a dead end
        neighbors = list(self.G.neighbors(self.state))
        if len(neighbors) == 0:
            return self.state, -1, True

        if self.reward == "unit":
            return self.state, 0, False
        elif self.reward == "weighted":
            return self.state, -(w + np.random.normal(scale=0.1)), False

    def path_cost(self, path):
        return nx.path_weight(self.G, path, "length")