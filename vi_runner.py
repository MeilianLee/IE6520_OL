import numpy as np
import itertools
import matplotlib.pyplot as plt


class Env:
    def __init__(self, landsize=3, total_year=10, water_capacity=10):
        self.target_habitat = 3
        self.target_water = 10
        self.landsize = landsize
        self.landuse = None
        self.water = None
        self.year = None
        self.habitat = 0
        self.total_year = total_year
        self.water_capacity = water_capacity

    def valid_actions(self, state):
        self.landuse = np.array(state[0])
        if 0 not in self.landuse:
            return [0]
        else:
            return [0, 1, 2, 3]

    def transit(self, state, action):
        """
        :param action: 0 - no-op, 1 - 1/3 hbt, 2 - 1/3 ofr, 3 - 1/3 wl
        :param state: (landuse, year, water)
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
                cost += 5 / 3
                self.habitat += 1
            elif action == 2:  # ofr
                cost += 4 / 3
            elif action == 3:  # wl
                cost += 7 / 3
            else:
                raise ValueError(f"Unknown action: {action}")

            self.landuse.sort()
        reward = -cost * 0.1

        self.year += 1
        if self.year >= self.total_year:
            done = True
            self.year = self.total_year
        else:
            done = False

        self.habitat = (self.landuse == 1).sum() + (self.landuse == 3).sum()
        if done:
            if self.habitat < self.target_habitat:
                reward -= 1
            else:
                reward += 1
            if self.water < self.target_water:
                reward -= 1
            else:
                reward += 1
        return (tuple(self.landuse), self.year, self.water), reward, done

    def calc_maintain_cost(self):
        cost_ag = - 0.05 / 3 * (self.landuse == 0).sum()
        cost_ofr = 0.2 / 3 * (self.landuse == 2).sum()
        cost_wl = 0.3 / 3 * (self.landuse == 3).sum()
        self.water += (self.landuse == 2).sum() + (self.landuse == 3).sum()
        self.water = min(self.water, self.water_capacity)
        return cost_ag + cost_ofr + cost_wl


class VIRunner:
    def __init__(self):
        self.landtype = {0: 'Ag', 1: 'Hbt', 2: 'Ofr', 3: 'Wl'}
        self.actions = [0, 1, 2, 3]
        self.landsize = 3
        self.horizon = 10
        self.max_water = 10
        self.env = Env(landsize=self.landsize, total_year=self.horizon, water_capacity=self.max_water)
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
        year = list(range(0, self.horizon+1))
        water = list(range(0, self.max_water+1))
        states = list(itertools.product(landuse, year, water))  # TODO: remove unreachable states
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
        eps = 0.01
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
        print(f"Converged at step {t}")
        self.gen_policy()
        self.eval_policy()


if __name__ == '__main__':
    runner = VIRunner()
    runner.run_vi()