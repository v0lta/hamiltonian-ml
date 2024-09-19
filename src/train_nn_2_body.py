import pickle
from sim_2_body import simulate, absolute_motion

import torch
import torch.nn as nn
import os.path
import numpy as np
from tqdm import tqdm
from networks import Feedforward
from networks import get_hamiltonian_2b as get_hamiltonian
from test_nn_2_body import test_net
from multiprocessing.pool import ThreadPool as Pool



def generate_data(iterations, batch_size, seed = 0):
    

    if os.path.isfile("data/twobody.pkl"):
        with open("data/twobody.pkl", "rb") as f:
            data, mean, std = pickle.load(f)
            return data, mean, std
    else:
        seed_offset = seed
        init = [np.array([1., 0.]),
                np.array([-1., 0.])]

        epoch_list = []
        for _ in tqdm(range(iterations), leave=False):
            # with Pool(4*5) as p:
            #     def sim_fun(s):
            #         return simulate(init, seed = s)
            #     data_epoch = p.map(sim_fun, np.arange(batch_size) + seed_offset - 1)
            data_epoch = []
            for seed in range(batch_size):
                orbit = simulate(init, seed = seed + seed_offset - 1)
                p1, p2, v1, v2, _, _, _ = orbit
                if orbit[-2] == True:
                    pq_vec = np.stack([p1, p2, v1, v2], 0)
                    pairs = (pq_vec[:, :, 0::2], pq_vec[:, :, 1::2])
                    pairs = np.stack(pairs, 0)
                    data_epoch.append(pairs)
            
            data_epoch = np.stack(data_epoch, 0)
            epoch_list.append(data_epoch.astype(np.float32))
            seed_offset += batch_size

        data = np.stack(epoch_list)
        mean = np.mean(data, axis=(0, 1, 2, 5)) 
        std = np.std(data, axis=(0, 1, 2, 5))

        mean = np.expand_dims(mean, (0, 1, 2, 5))
        std = np.expand_dims(std, (0, 1, 2, 5))

        with open("data/twobody.pkl", "wb") as f:
            pickle.dump([data, mean, std], f)
        return data, mean, std




if __name__ == '__main__':
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    epochs = 10
    batch_size = 200
    iterations = 10
    conserve = True
    vars = 4
    dims = 2
    dt = 0.05
    in_size = dims*vars
    out_size = 1 if conserve else in_size

    data, mean, std = generate_data(iterations, batch_size, seed=0)


    # data = (data - mean)/std
    print(f"Conserve: {conserve}")
    print(f"data mean {np.mean(data)}, data std {np.std(data)}")
    print(f"data shape {data.shape}")

    network = Feedforward(in_size, 200, out_size).cuda()
    opt = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fun = torch.nn.MSELoss()

    print(f"energy conservation: {conserve}")

    test_loss = 'Not available'

    for e in tqdm(range(epochs), desc="Training Network"):
        epoch_bar = tqdm(range(iterations), desc=f"Epoch progress, last test loss {test_loss}", leave=False)
        for i in epoch_bar:
            rng.shuffle(data, axis=-1)
            bar = tqdm(range(data.shape[-1]), desc="Time Loop", leave=False)
            for t in bar:
                xy = data[i, :, :, :, :, t]
                input_x = torch.tensor(xy[:, 0, :, :], requires_grad=True).cuda()
                # input_x += torch.randn_like(input_x)*0.01
                output_y = torch.tensor(xy[:, 1, :, :]).cuda()

                input_x = input_x*0 + torch.tensor([[1, 1], [-1, -1], [-0, 0], [-0, 0]]).unsqueeze(0).cuda()
                output_y = input_x

                if conserve:
                    loss, dx_dt_hat, dx_dt = get_hamiltonian(input_x, output_y, network)
                else:
                    dx_dt = torch.tensor(absolute_motion(None, input_x.detach().cpu().numpy().swapaxes(0, 1))
                                         ).to(input_x.device).swapdims(0, 1)
                    dx_dt_hat = network(input_x)
                    dx_dt_hat = torch.reshape(dx_dt_hat, [batch_size, 4, 2])
                    loss = loss_fun(dx_dt_hat, dx_dt)
                    conservation_loss = torch.tensor(0.)

                loss.backward()
                opt.step()

                y_hat =  input_x + dt*dx_dt_hat
                pred_loss = loss_fun(output_y, y_hat)

                dploss = pred_loss.detach().cpu().numpy()
                dloss = loss.detach().cpu().numpy()

                grad = torch.cat([param.grad.flatten() for param in network.parameters()]).clone()
                opt.zero_grad()
                bar.set_description(f"Prediction loss: {dploss:.4e}, Energy: {dloss:.4e}, grad-norm: {torch.linalg.norm(grad):.4e}, grad-std: {torch.linalg.norm(grad):.4e}")
            network.cpu()
            if e % 1 == 0:
                test_loss, _, _ = test_net(network, seed=-1)
                epoch_bar.set_description(f"Epoch progress, test-loss: {float(test_loss):.4e}")
                torch.save([network, mean, std], f"network_c{conserve}.pt")
            network.cuda()
            

    print("training done")
    network.cpu()
    torch.save([network, mean, std], f"network_c{conserve}.pt")

    pass