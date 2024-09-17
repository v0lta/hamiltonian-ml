import pickle
from sim_2_body import simulate

import torch
import torch.nn as nn
import os.path
import numpy as np
from tqdm import tqdm
from networks import Feedforward
from test_nn_2_body import test_net
from sim_2_body import absolute_motion



def generate_data(epoch, iterations, batch_size, seed = 0):
    

    if os.path.isfile("data/twobody.pkl"):
        with open("data/twobody.pkl", "rb") as f:
            data = pickle.load(f)
            return data
    else:
        G=1.
        m1=1.
        m2=1.
        std = 0.0025
        t_max = 600
        seed_offset = 0

        init = [np.array([0., 0.]),
                np.array([0., .97]),
                np.array([1., 0.]),
                np.array([-1., 0.])]
        data = []
        for _ in tqdm(range(epoch), desc="Simulating two-body problem."):
            epoch_list = []
            for _ in tqdm(range(iterations), leave=False):
                data_epoch = [simulate(init, seed = seed + seed_offset - 1, std = std, G=G, m1=m1, m2=m2, t_max=t_max) for seed in range(batch_size)]
                data_epoch = list(filter(lambda r: r[-1] == True, data_epoch))
                # p1, p2, v1, v2, t_points, _ = r
                data_epoch_clean = [(de[0], de[1], de[2], de[3]) for de in data_epoch]
                data_epoch_clean = np.stack(data_epoch_clean, 1)
                nskip = 1
                data_epoch_clean_subsample = data_epoch_clean[:, :, :, ::nskip]
                epoch_list.append(data_epoch_clean_subsample.astype(np.float32))
                seed_offset += batch_size
            data.append(epoch_list)
            
        with open("data/twobody.pkl", "wb") as f:
            pickle.dump(data, f)
        return data


def get_hamiltonian(in_x, network, motion_fun):
    

    dx_dt_hat = network.dxdt(in_x)
    dx_dt = torch.tensor(motion_fun(in_x.detach().cpu().numpy())).cuda()
    
    dh_dx = torch.autograd.grad(dx_dt_hat.sum(), in_x, create_graph=True)[0]
    dh_dp, dh_dq = torch.split(dh_dx, 2, dim=1)

    # dxdt = (output_y - input_x)/0.5
    # p1, p2, v1, v2
    loss_fun = torch.nn.MSELoss()
    dp_dt, dq_dt = torch.split(dx_dt, 2, dim=1)
    conservation_loss = loss_fun(dq_dt, -dh_dp) + loss_fun(dp_dt, dh_dq)
                
    return conservation_loss, dx_dt_hat, dx_dt



if __name__ == '__main__':
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    epochs = 20
    batch_size = 200
    iterations = 15
    dt = 0.5
    
    data = generate_data(epochs, iterations, batch_size, seed=0)

    network = Feedforward(2*4, 512, 2*4).cuda()
    opt = torch.optim.Adam(network.parameters(), lr=1e-3)
    # opt = torch.optim.rmsprop(network.parameters(), lr=1e-4)
    loss_fun = torch.nn.MSELoss()
    conserve = False
    print(f"energy conservation: {conserve}")

    for e in tqdm(range(epochs), desc="Training Network"):
        epoch_data = data[e]
        epoch_bar = tqdm(range(iterations), desc="Epoch progress", leave=False)
        for i in epoch_bar:
            time_pairs = [(t, t+1) for t in range(epoch_data[i].shape[-1] - 1)]
            rng.shuffle(time_pairs)
            bar = tqdm(time_pairs, desc="Time Loop", leave=False)
            for t, tpone in bar:
                input_x = torch.tensor(epoch_data[i][:, :, :, t].swapaxes(0, 1), requires_grad=True).cuda()
                input_x.retain_grad()
                output_y = torch.tensor(epoch_data[i][:, :, :, tpone].swapaxes(0, 1)).cuda()

                opt.zero_grad()
                if input_x.grad:
                    input_x.grad.zero_()

                motion_fun = lambda x: absolute_motion(None, x)
                conservation_loss, dx_dt_hat, dx_dt = get_hamiltonian(input_x, network, motion_fun)

                pred_loss = loss_fun(dx_dt_hat, dx_dt)

                if conserve:
                    loss = pred_loss * 0.1*conservation_loss
                else:
                    y_hat = input_x + dt*dx_dt_hat
                    pred_loss = loss_fun(output_y, y_hat)
                    loss = pred_loss
                loss.backward()
                opt.step()
                dloss = pred_loss.detach().cpu().numpy()
                deng = conservation_loss.detach().cpu().numpy()

                grad = torch.cat([param.grad.flatten() for param in network.parameters()]).clone()

                bar.set_description(f"Prediction loss: {dloss:3.8f}, Energy: {deng:3.8f}, Grad-Norm: {torch.linalg.norm(grad):2.6f}")
            network.cpu()
            test_loss, _, _ = test_net(network, 2)
            epoch_bar.set_description(f"Epoch progress, test-loss: {float(test_loss):2.2f}")
            torch.save(network, f"network_c{conserve}.pt")
            network.cuda()
            

    print("training done")
    network.cpu()
    torch.save(network, f"network_c{conserve}.pt")

    pass