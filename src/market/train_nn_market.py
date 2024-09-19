from sim_market import market_sim, normalize

import torch

import numpy as np
from tqdm import tqdm
from networks import Feedforward


def generate_data(iterations, batch_size, seed = 0):
    
        t = 100
        dt = 0.1
        time = np.arange(0, t, dt)
        exp_return = np.array([.1, 0., -.1])
        sigma = 0.8

        start_price = normalize(np.array([5, 4, 3]))
        sim = market_sim(start_price, exp_return, time)

        return None




if __name__ == '__main__':
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    pass