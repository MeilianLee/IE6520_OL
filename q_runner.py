import numpy as np
import itertools
from arguments import arg
from env import Env


class QLearner:
    def __init__(self):
        self.landtype = {0: 'Ag', 1: 'Ofr', 2: 'Wl'}
        self.landsize = arg.landsize
        self.horizon = arg.horizon
        self.max_water = 3
        self.env = Env(landsize=self.landsize, total_year=self.horizon)
        self.q_table = {}
        self.pi_table = {}
        self.n_episode = int(1e5)
        self.lr = 0.01
        self.gamma = 0.99
        self.epsilon = 0.3
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

    def eval_policy(self, eval=True):
        all_states = [self.env.reset()]
        all_actions = []
        all_reward = 0
        while True:
            action = self.find_rand_max_idx(self.pi_table[all_states[-1]])
            valid_actions = self.env.valid_actions(all_states[-1])
            if action not in valid_actions:
                action = 0
            state, reward, done = self.env.step(all_states[-1], action, eval=eval)
            all_actions += [action]
            all_states += [state]
            all_reward += reward
            if done:
                break
        print(f"Actions: {all_actions}")
        print(f"Reward: {all_reward}")
        print(f"Landuse: {self.env.landuse}, {(self.env.landuse > 1).sum() >= self.env.target_habitat}")
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
            if episode % 1000 == 0:
                print(f"Episode {episode} finished in {step} steps")
                self.eval_policy()


if __name__ == '__main__':
    runner = QLearner()
    runner.run_qlearning()
