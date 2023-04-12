import numpy as np
import itertools
import matplotlib.pyplot as plt


class Env:
    def __init__(self, landsize=3):
        self.landsize = landsize
        self.landuse = np.zeros(landsize, dtype=int)
        self.water = 0
        self.habitat = 0
        self.year = 0
        self.total_year = 20

    def reset(self):
        pass

    def step(self, action):
        """
        :param action: 0 - no-op, 1 - 1/3 hbt, 2 - 1/3 ofr, 3 - 1/3 wl
        :return: next_state, reward, done
        """
        cost = self.calc_maintain_cost()
        if action == 0:
            pass
        else:
            assert 0 in self.landuse, f"No empty land: {self.landuse}"
            self.landuse[np.argwhere(self.landuse == 0)[0]] = action
            if action == 1:  # hbt
                cost += 2
                self.habitat += 1
            elif action == 2:  # ofr
                cost += 1
            elif action == 3:  # wl
                cost += 3
                self.habitat += 1
            else:
                raise ValueError(f"Unknown action: {action}")

            self.landuse.sort()
        self.year += 1
        reward = -cost
        done = self.year == self.total_year
        if done:
            if self.habitat >= 2:
                reward += 5
            else:
                reward -= 5
            if self.water >= 2:
                reward += 5
            else:
                reward -= 5
        return self.landuse, reward, done

    def calc_maintain_cost(self):
        cost_ag = -0.05 * (self.landuse == 0).sum()
        cost_ofr = 0.1 * (self.landuse == 2).sum()
        cost_wl = 0.1 * (self.landuse == 3).sum()
        self.water += 0.1 * ((self.landuse == 2).sum() + (self.landuse == 3).sum())
        return cost_ag + cost_ofr + cost_wl



class VIRunner:
    def __init__(self):
        self.landtype = {0: 'Ag', 1: 'Hbt', 2: 'Ofr', 3: 'Wl'}
        self.actions = [0, 1, 2, 3]
        self.landsize = 3
        self.env = Env(landsize=self.landsize)
        self.horizon = 20
        self.states = list(itertools.combinations_with_replacement(self.landtype.keys(), self.landsize))
        self.state_values = {}
        self.advantage = {}
        self.diff_values = {}
        self.optim = None
        self.step = 0

    def init_states(self):
        self.state_values = {vs: [] for vs in self.states}
        self.advantage = {vs: None for vs in self.states}
        self.diff_values = {vs: None for vs in self.states}

    def run_vi(self):
        eps = 1e-3
        gap = np.inf
        while gap > eps:
            for state in self.states:



if __name__ == '__main__':
    pass