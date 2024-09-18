import pickle
from sim_2_body import simulate, absolute_motion

import torch
import torch.nn as nn
import os.path
import numpy as np
from tqdm import tqdm
from networks import Feedforward, get_hamiltonian
from test_nn_2_body import test_net
from multiprocessing.pool import ThreadPool as Pool



def generate_data(iterations, batch_size, seed = 0):
    

    if os.path.isfile("data/twobody.pkl"):
        with open("data/twobody.pkl", "rb") as f:
            data, mean, std = pickle.load(f)
            return data, mean, std
    else:
        seed_offset = seed
        init = [np.array([2, 0.]),
                np.array([-2, 0.])]

        data = []
        epoch_list = []
        for _ in tqdm(range(iterations), leave=False):
            # with Pool(4*5) as p:
            #     def sim_fun(s):
            #         return simulate(init, seed = s)
            #     data_epoch = p.map(sim_fun, np.arange(batch_size) + seed_offset - 1)
            data_epoch = [simulate(init, seed = seed + seed_offset - 1) for seed in range(batch_size)]
            data_epoch = list(filter(lambda r: r[-2] == True, data_epoch))
            # p1, p2, v1, v2, t_points, _ = r
            data_epoch_clean = [(de[0], de[1], de[2], de[3]) for de in data_epoch]
            data_epoch_clean = np.stack(data_epoch_clean, 1)
            nskip = 1
            data_epoch_clean_subsample = data_epoch_clean[:, :, :, ::nskip]
            epoch_list.append(data_epoch_clean_subsample.astype(np.float32))
            seed_offset += batch_size

        data = np.stack(epoch_list)
        mean = np.mean(data, axis=(0, 2, 4)) 
        std = np.std(data, axis=(0, 2, 4))

        mean = np.expand_dims(mean, (0, 2, 4))
        std = np.expand_dims(std, (0, 2, 4))

        with open("data/twobody.pkl", "wb") as f:
            pickle.dump([data, mean, std], f)
        return data, mean, std




if __name__ == '__main__':
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    epochs = 20
    batch_size = 200
    iterations = 200
    conserve = True
    vars = 4
    dims = 2
    dt = 0.5
    in_size = dims*vars
    out_size = 1 if conserve else in_size

    data, mean, std = generate_data(iterations, batch_size, seed=0)
    mean = mean*0.
    std = std*0 + 1.

    # data = (data - mean)/std
    print(f"Conserve: {conserve}")
    print(f"data mean {np.mean(data)}, data std {np.std(data)}")
    print(f"data shape {data.shape}")

    network = Feedforward(in_size, 512, out_size, mean, std).cuda()
    opt = torch.optim.Adam(network.parameters(), lr=1e-5)
    loss_fun = torch.nn.MSELoss()

    print(f"energy conservation: {conserve}")

    test_loss = 'Not available'

    for e in tqdm(range(epochs), desc="Training Network"):
        epoch_bar = tqdm(range(iterations), desc=f"Epoch progress, last test loss {test_loss}", leave=False)
        for e in epoch_bar:
            time_pairs = [(t, t+1) for t in range(data[e].shape[-1] - 1)]
            rng.shuffle(time_pairs)
            bar = tqdm(time_pairs, desc="Time Loop", leave=False)
            for t, tpone in bar:
                input_x = torch.tensor(data[e][:, :, :, t].swapaxes(0, 1), requires_grad=True).cuda()
                input_x.retain_grad()
                output_y = torch.tensor(data[e][:, :, :, tpone].swapaxes(0, 1)).cuda()
                opt.zero_grad()

                input_x = input_x + torch.randn_like(input_x)*0.001


                if conserve:
                    conservation_loss, dx_dt_hat, dx_dt = get_hamiltonian(input_x, output_y, network)
                    loss = conservation_loss
                else:
                    dx_dt = torch.tensor(absolute_motion(None, input_x.detach().cpu().numpy().swapaxes(0, 1))
                                         ).to(input_x.device).swapdims(0, 1)
                    dx_dt_hat = network(input_x)
                    dx_dt_hat = torch.reshape(dx_dt_hat, [batch_size, 4, 2])
                    direct_loss = loss_fun(dx_dt_hat, dx_dt)
                    loss = direct_loss
                    conservation_loss = torch.tensor(0.)

                y_hat =  input_x + dt*dx_dt_hat
                pred_loss = loss_fun(output_y, y_hat)

                loss.backward()
                opt.step()
                dloss = pred_loss.detach().cpu().numpy()
                deng = conservation_loss.detach().cpu().numpy()

                grad = torch.cat([param.grad.flatten() for param in network.parameters()]).clone()

                bar.set_description(f"Prediction loss: {dloss:3.8f}, Energy: {deng:3.8f}, Grad-Norm: {torch.linalg.norm(grad):2.6f}")
            network.cpu()
            test_loss, _, _ = test_net(network, seed=-1, data_mean=mean, data_std=std)
            epoch_bar.set_description(f"Epoch progress, test-loss: {float(test_loss):2.2f}")
            torch.save([network, mean, std], f"network_c{conserve}.pt")
            network.cuda()
            

    print("training done")
    network.cpu()
    torch.save([network, mean, std], f"network_c{conserve}.pt")

    pass