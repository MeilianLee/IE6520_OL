import numpy as np
import itertools
import matplotlib.pyplot as plt


class Env:
    def __init__(self, landsize=3):
        self.landsize = landsize
        self.landuse = None
        self.water = None
        self.year = None
        self.habitat = 0
        self.total_year = 10

    def valid_actions(self, state):
        self.landuse = np.array(state[0])
        if 0 not in self.landuse:
            return [0]
        else:
            return [0, 1, 2, 3]

    def transit(self, state, action):
        """
        :param action: 0 - no-op, 1 - 1/3 hbt, 2 - 1/3 ofr, 3 - 1/3 wl
        :return: next_state, reward, done
        """
        self.landuse = np.array(state[0])
        self.year = state[1]
        self.water = state[2]
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
        reward = -cost
        done = self.year >= self.total_year
        if done:
            if self.habitat >= 2:
                reward += 5
            else:
                reward -= 5
            if self.water >= 1:
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
        self.horizon = 10
        self.states = self.init_states()
        self.state_values = {}
        self.policy = {}
        self.advantage = {}
        self.diff_values = {}
        self.optim = None
        self.step = 0

    def init_states(self):
        landuse = list(itertools.combinations_with_replacement(self.landtype.keys(), self.landsize))
        year = list(range(1, self.horizon+1))
        water = np.arange(0, 1.1, 0.1)
        states = list(itertools.product(landuse, year, water))
        self.state_values = {s: [0] for s in states}
        self.policy = {s: None for s in states}
        self.advantage = {s: None for s in states}
        self.diff_values = {s: None for s in states}
        return states

    def run_vi(self):
        eps = 1e-3
        gap = np.inf
        t = 0
        while gap > eps:
            for state in self.states:
                valid_actions = self.env.valid_actions(state)



if __name__ == '__main__':
    pass