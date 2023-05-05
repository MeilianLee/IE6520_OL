import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt


#%%
class Env:
    def __init__(self, landsize=3, total_year=10):
        self.target_habitat = 0.7
        self.target_water = 0.1
        self.landsize = landsize
        self.landuse = None
        self.water = 0
        self.water_state = None
        self.year = None
        self.habitat = 0
        self.total_year = total_year
        self.supply = pd.read_csv('water_supply.csv', header=None).values.flatten()

    def reset(self):
        self.landuse = np.zeros(self.landsize, dtype=int)
        self.water = 0
        self.year = 0
        self.habitat = 0
        self.water_state = 0
        return tuple(self.landuse), self.year, self.water_state

    def valid_actions(self, state):
        self.landuse = np.array(state[0])
        if 0 not in self.landuse:
            return [0]
        else:
            return [0, 1, 2]

    def convert_water_state(self):
        if self.water <= 0:
            return 0
        elif self.water < 0.04:
            return 1
        elif self.water < 0.08:
            return 2
        elif self.water >= 0.08:
            return 3

    def step(self, state, action):
        """
        :param action: 0 - no-op, 1 - 1/3 ofr, 2 - 1/3 wl
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
            if action == 1:  # ofr
                cost += 0.27 / self.landsize
            elif action == 2:  # wl
                cost += 0.165 / self.landsize
            else:
                raise ValueError(f"Unknown action: {action}")

            self.landuse.sort()
        reward = -cost

        self.year += 1
        if self.year >= self.total_year:
            done = True
            self.year = self.total_year
        else:
            done = False

        self.habitat = (self.landuse > 0).sum() / self.landsize
        if done:
            if self.habitat < self.target_habitat:
                reward -= 1
            if self.water < self.target_water:
                reward -= 1
        return (tuple(self.landuse), self.year, self.water_state), reward, done

    def calc_maintain_cost(self):
        rev_ag = - 0.2 / self.landsize * (self.landuse == 0).sum()
        rev_ofr = - 0.2 / self.landsize * (self.landuse == 1).sum()
        cost_ofr = 0.1 / self.landsize * (self.landuse == 1).sum()
        cost_wl = 0.15 / self.landsize * (self.landuse == 2).sum()
        recharge_capacity_ofr = 0.085 / self.landsize
        recharge_capacity_wl = 0.2 / self.landsize
        demand = 0.381
        # read csv of water supply (20 years) and use self.year to index
        d_water = (self.supply[self.year-1] - demand) / self.landsize
        if d_water > 0:
            # compare with recharge capacity
            recharge_volume_ofr = min(recharge_capacity_ofr, d_water)
            pump_volume = 0
        else:
            recharge_volume_ofr = 0
            pump_volume = -d_water
        recharge_volume_wl = min(recharge_capacity_wl, self.supply[self.year - 1] / self.landsize)
        self.water = (self.landuse == 1).sum() * recharge_volume_ofr + (self.landuse == 2).sum() * recharge_volume_wl - \
                     ((self.landuse == 0) | (self.landuse == 1)).sum() * pump_volume
        pump_cost = 10 * pump_volume * ((self.landuse == 0) | (self.landuse == 1)).sum()
        # if pump_cost != 0:
        #     print(f"pump cost: {pump_cost}", rev_ag, cost_ofr, cost_wl)

        return rev_ag + rev_ofr + cost_ofr + cost_wl + pump_cost


class QLearner:
    def __init__(self):
        self.landtype = {0: 'Ag', 1: 'Ofr', 2: 'Wl'}
        self.landsize = 5
        self.horizon = 10
        self.max_water = 3
        self.env = Env(landsize=self.landsize, total_year=self.horizon)
        self.q_table = {}
        self.pi_table = {}
        self.n_episode = 1000
        self.lr = 0.01
        self.gamma = 0.95
        self.epsilon = 0.1
        self.states = self.init_states()

    def init_states(self):
        landuse = list(itertools.combinations_with_replacement(self.landtype.keys(), self.landsize))
        year = list(range(0, self.horizon+1))
        water = list(range(0, self.max_water+1))
        states = list(itertools.product(landuse, year, water))
        for s in states:
            n_a = len(self.env.valid_actions(s))
            self.q_table[s] = [0] * n_a
            self.pi_table[s] = [1 / n_a] * n_a
        return states

    @staticmethod
    def find_rand_max_idx(qs):
        idx_list = []
        for idx, q in enumerate(qs):
            if q == max(qs):
                idx_list.append(idx)
        action = np.random.choice(idx_list)
        return action

    def eval_policy(self):
        all_states = [self.env.reset()]
        all_actions = []
        all_reward = 0
        while True:
            action = np.argmax(self.pi_table[all_states[-1]])
            valid_actions = self.env.valid_actions(all_states[-1])
            if action not in valid_actions:
                action = 0
            state, reward, done = self.env.step(all_states[-1], action)
            all_actions += [action]
            all_states += [state]
            all_reward += reward
            if done:
                break
        print(f"Actions: {all_actions}")
        print(f"Reward: {all_reward}")
        print(f"Landuse: {self.env.landuse}, {(self.env.landuse > 0).sum() >= self.env.target_habitat}")
        print(f"Water: {self.env.water}, {self.env.water >= self.env.target_water}")
        print("=====================================")


    def run_qlearning(self):
        for episode in range(self.n_episode):
            step = 0
            done = False
            state = self.env.reset()
            action = np.random.choice(self.env.valid_actions(state), p=self.pi_table[state])
            while not done:
                step += 1
                new_state, reward, done = self.env.step(state, action)
                new_action = np.random.choice(self.env.valid_actions(new_state), p=self.pi_table[new_state])
                new_q = max(self.q_table[new_state])  # Q-learning
                # new_q = self.q_table[new_state][new_action]  # SARSA
                self.q_table[state][action] += self.lr * (reward + self.gamma * new_q - self.q_table[state][action])
                best_action = np.argmax(self.q_table[state])
                for a in range(len(self.q_table[state])):
                    if a == best_action:
                        self.pi_table[state][a] = 1 - self.epsilon + self.epsilon / len(self.q_table[state])
                    else:
                        self.pi_table[state][a] = self.epsilon / len(self.q_table[state])
                state = new_state
                action = new_action
            if episode % 100 == 0:
                print(f"Episode {episode} finished in {step} steps")
                self.eval_policy()



if __name__ == '__main__':
    runner = QLearner()
    runner.run_qlearning()
