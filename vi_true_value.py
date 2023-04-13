import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt


#%%
class Env:
    def __init__(self, landsize=3, total_year=10):
        self.target_habitat = 1
        self.target_water = 0.12
        self.landsize = landsize
        self.landuse = None
        self.water = 0
        self.water_state = None
        self.year = None
        self.habitat = 0
        self.total_year = total_year
        self.supply = pd.read_csv('water_supply.csv', header=None).values.flatten()
        # self.water_capacity = water_capacity

    def valid_actions(self, state):
        self.landuse = np.array(state[0])
        if 0 not in self.landuse:
            return [0]
        else:
            return [0, 1, 2, 3]

    def convert_water_state(self):
        if self.water <= 0:
            return 0
        elif self.water < 0.04:
            return 1
        elif self.water < 0.08:
            return 2
        elif self.water >= 0.08:
            return 3

    def transit(self, state, action):
        """
        :param action: 0 - no-op, 1 - 1/3 hbt, 2 - 1/3 ofr, 3 - 1/3 wl
        :return: next_state, reward, done
        """
        self.landuse = np.array(state[0])
        self.year = state[1]
        # self.water_state = state[2]
        cost = self.calc_maintain_cost()
        self.water_state = self.convert_water_state()
        if action == 0:
            pass
        else:
            assert 0 in self.landuse, f"No empty land: {self.landuse}"
            self.landuse[np.argwhere(self.landuse == 0)[0]] = action
            if action == 1:  # hbt
                cost += 0.015
                self.habitat += 1
            elif action == 2:  # ofr
                cost += 0.09
            elif action == 3:  # wl
                cost += 0.055
                self.habitat += 1
            else:
                raise ValueError(f"Unknown action: {action}")

            self.landuse.sort()
        reward = -cost
        done = self.year >= self.total_year
        self.year = self.year + 1 if not done else self.year
        self.habitat = (self.landuse > 0).sum()
        if done:
            if self.habitat < self.target_habitat:
                reward -= 1
            else:
                reward += 1
            if self.water < self.target_water:
                reward -= 1
            else:
                reward += 1
            self.water = 0
        return (tuple(self.landuse), self.year, self.water_state), reward, done

    def calc_maintain_cost(self):
        rev_ag = - 0.07 * ((self.landuse == 0) | (self.landuse == 2)).sum()
        cost_ofr = 0.034 * (self.landuse == 2).sum()
        cost_wl = 0.045 * (self.landuse == 3).sum()
        # self.water += (self.landuse == 2).sum() + (self.landuse == 3).sum()
        recharge_capacity_Ofr = 0.085 / 3
        recharge_capacity_Wl = 0.2 / 3
        demand = 0.381
        # read csv of water supply (20 years) and use self.year to index
        if self.supply[self.year-1]/3 - demand/3 > 0:
            # compare with recharge capacity
            recharge_volume_Ofr = min(recharge_capacity_Ofr, self.supply[self.year-1]/3 - demand/3)
            pump_volume = 0
        else:
            recharge_volume_Ofr = 0
            pump_volume = demand/3 - self.supply[self.year-1]/3
        recharge_volume_Wl = min(recharge_capacity_Wl, self.supply[self.year-1]/3)
        self.water += (self.landuse == 2).sum() * recharge_volume_Ofr + (self.landuse == 3).sum() * recharge_volume_Wl - \
                      (self.landuse == 0).sum() * pump_volume
        pump_cost = 0.01 * pump_volume * ((self.landuse == 0) | (self.landuse == 2)).sum()

        # self.water = min(self.water, self.water_capacity)
        return rev_ag + cost_ofr + cost_wl + pump_cost


class VIRunner:
    def __init__(self):
        self.landtype = {0: 'Ag', 1: 'Hbt', 2: 'Ofr', 3: 'Wl'}
        self.actions = [0, 1, 2, 3]
        self.landsize = 3
        self.horizon = 10
        self.max_water = 3
        self.env = Env(landsize=self.landsize, total_year=self.horizon)
        self.state_values = {}
        self.new_state_values = {}
        self.policy = {}
        self.advantage = {}
        self.diff_values = {}
        self.optim = None
        self.step = 0
        self.states = self.init_states()

    def init_states(self):
        landuse = list(itertools.combinations_with_replacement(self.landtype.keys(), self.landsize))
        year = list(range(1, self.horizon+1))
        water = list(range(0, self.max_water+1))
        states = list(itertools.product(landuse, year, water))
        n_states = len(states)
        self.state_values = np.zeros(n_states)
        self.new_state_values = np.zeros(n_states)
        self.policy = np.empty(n_states, dtype=int)
        self.advantage = np.empty(n_states)
        self.diff_values = np.empty(n_states)
        return states

    def find_idx(self, state):
        idx = self.states.index(state)
        return idx

    def gen_policy(self):
        for state in self.states:
            valid_actions = self.env.valid_actions(state)
            next_state_values = []
            idx = self.find_idx(state)
            for action in valid_actions:
                next_state, reward, _ = self.env.transit(state, action)
                next_idx = self.find_idx(next_state)
                next_state_value = reward + self.state_values[next_idx]
                next_state_values += [next_state_value]
            self.policy[idx] = valid_actions[np.argmax(next_state_values)]
            # self.advantage[idx] = self.state_values[idx] - np.max(next_state_values)

    def eval_policy(self):
        idx = 0
        all_states = [self.states[idx]]
        all_actions = []
        all_reward = 0
        while True:
            action = self.policy[idx]
            valid_actions = self.env.valid_actions(all_states[-1])
            if action not in valid_actions:
                action = 0
            state, reward, done = self.env.transit(all_states[-1], action)
            idx = self.find_idx(state)
            all_actions += [action]
            all_states += [state]
            all_reward += reward
            if done:
                break
        print(f"States: {all_states}")
        print(f"Actions: {all_actions}")
        print(f"Reward: {all_reward}")


    def run_vi(self):
        eps = 0.1
        gap = np.inf
        t = 0
        while gap > eps:
            if t % 1 == 0:
                self.gen_policy()
                self.eval_policy()
            for state in self.states:
                valid_actions = self.env.valid_actions(state)
                next_state_values = []
                idx = self.find_idx(state)
                for action in valid_actions:
                    next_state, reward, done = self.env.transit(state, action)
                    next_idx = self.find_idx(next_state)
                    next_state_value = reward + self.state_values[next_idx]
                    next_state_values += [next_state_value]
                self.new_state_values[idx] = np.max(next_state_values)
            self.diff_values = self.new_state_values - self.state_values
            self.state_values = self.new_state_values.copy()
            t += 1
            gap = self.diff_values.max() - self.diff_values.min()
            avg_gap = self.diff_values.mean()
            avg_value = self.state_values.mean()
            print(f"Step {t}: gap = {gap:.6f}, avg_gap = {avg_gap:.6f}, avg_value = {avg_value:.6f}, policy = {self.policy.sum()}")
            # print(f"Water: {self.env.water: .4f}")
        print(f"Converged at step {t}")
        self.gen_policy()
        self.eval_policy()


if __name__ == '__main__':
    runner = VIRunner()
    runner.run_vi()
