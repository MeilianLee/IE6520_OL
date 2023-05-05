import numpy as np
import itertools
from arguments import arg
from env import Env


class VIRunner:
    def __init__(self):
        self.landtype = {0: 'Ag', 1: 'Ofr', 2: 'Wl'}
        self.landsize = arg.landsize
        self.horizon = arg.horizon
        self.max_water = 3
        self.env = Env(landsize=self.landsize, total_year=self.horizon)
        self.state_values = None
        self.new_state_values = None
        self.policy = None
        self.advantage = None
        self.diff_values = None
        self.step = 0
        self.n_episode = 100
        self.states = self.init_states()

    def init_states(self):
        landuse = list(itertools.combinations_with_replacement(self.landtype.keys(), self.landsize))
        year = list(range(0, self.horizon+1))
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
                next_state, reward, _ = self.env.step(state, action)
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
            state, reward, done = self.env.step(all_states[-1], action, eval=True)
            idx = self.find_idx(state)
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


    def run_vi(self):
        eps = 0.1
        gap = np.inf
        t = 0
        while gap > eps and t < self.n_episode:
            if t % 10 == 0:
                self.gen_policy()
                self.eval_policy()
            for state in self.states:
                valid_actions = self.env.valid_actions(state)
                next_state_values = []
                idx = self.find_idx(state)
                for action in valid_actions:
                    next_state, reward, done = self.env.step(state, action)
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
        # print(f"Converged at step {t}")
        self.gen_policy()
        self.eval_policy()


if __name__ == '__main__':
    runner = VIRunner()
    runner.run_vi()
