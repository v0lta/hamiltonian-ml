from sim_market import market_sim, normalize

import torch

import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from tqdm import tqdm
from networks import Feedforward


def generate_data(batch_size, rng):
    
        t = 100
        dt = 0.1
        time = np.arange(0, t, dt)
        exp_return = np.array([.1, 0.])

        start_price = normalize(np.array([5, 4]))
        sim = [market_sim(start_price, exp_return, time, rng) for _ in range(batch_size)]

        market, dxdts = zip(*sim)
        market = np.stack(market)
        dxdts = np.stack(dxdts)

        return market, dxdts



if __name__ == '__main__':
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    batch_size = 64

    market, dxdts = generate_data(batch_size=batch_size, rng=rng)
    print(market.shape)

    pass