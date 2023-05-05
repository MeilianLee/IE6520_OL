import numpy as np
import pandas as pd
from arguments import arg


class Env:
    def __init__(self, landsize=3, total_year=10):
        self.target_habitat = arg.target_habitat
        self.target_water = arg.target_water
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

    def step(self, state, action, eval=False):
        """
        :param action: 0 - no-op, 1 - 1/3 ofr, 2 - 1/3 wl
        :return: next_state, reward, done
        """
        self.landuse = np.array(state[0])
        self.year = state[1]
        # self.water_state = state[2]
        cost = self.calc_maintain_cost(eval)
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

    def calc_maintain_cost(self, eval=False):
        rev_ag = - 0.2 / self.landsize * (self.landuse == 0).sum()
        rev_ofr = - 0.2 / self.landsize * (self.landuse == 1).sum()
        cost_ofr = 0.1 / self.landsize * (self.landuse == 1).sum()
        cost_wl = 0.15 / self.landsize * (self.landuse == 2).sum()
        recharge_capacity_ofr = 0.085 / self.landsize
        recharge_capacity_wl = 0.2 / self.landsize
        demand = 0.381
        supply = np.random.normal(0.391, 0.01) if arg.random_supply and not eval else self.supply[self.year-1]
        # read csv of water supply (20 years) and use self.year to index
        d_water = (supply - demand) / self.landsize
        if d_water > 0:
            # compare with recharge capacity
            recharge_volume_ofr = min(recharge_capacity_ofr, d_water)
            pump_volume = 0
        else:
            recharge_volume_ofr = 0
            pump_volume = -d_water
        recharge_volume_wl = min(recharge_capacity_wl, supply / self.landsize)
        self.water = (self.landuse == 1).sum() * recharge_volume_ofr + (self.landuse == 2).sum() * recharge_volume_wl - \
                     ((self.landuse == 0) | (self.landuse == 1)).sum() * pump_volume
        pump_cost = 10 * pump_volume * ((self.landuse == 0) | (self.landuse == 1)).sum()
        # if pump_cost != 0:
        #     print(f"pump cost: {pump_cost}", rev_ag, cost_ofr, cost_wl)

        return rev_ag + rev_ofr + cost_ofr + cost_wl + pump_cost