import pickle
from sim_2_body import simulate

import torch
import os.path
import numpy as np
from tqdm import tqdm
from networks import Feedforward
from test_nn_2_body import test_net



def generate_data(epoch, iterations, batch_size, seed = 0):
    

    if os.path.isfile("data/twobody.pkl"):
        with open("data/twobody.pkl", "rb") as f:
            data = pickle.load(f)
            return data
    else:
        G=1.
        m1=1.
        m2=1.
        std = 0.005
        t_max = 800
        seed_offset =0

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
                data_epoch_clean_subsample = data_epoch_clean[:, :, :, ::50]
                epoch_list.append(data_epoch_clean_subsample.astype(np.float32))
                seed_offset += batch_size
            data.append(epoch_list)
            
        with open("data/twobody.pkl", "wb") as f:
            pickle.dump(data, f)
        return data




if __name__ == '__main__':
    torch.manual_seed(42)

    epochs = 10
    batch_size = 200
    iterations = 15
    
    data = generate_data(epochs, iterations, batch_size, seed=0)

    network = Feedforward(2*4, 200, 2*4)
    opt = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fun = torch.nn.MSELoss()
    conserve = True

    for e in tqdm(range(epochs), desc="Training Network"):
        epoch_data = data[e]
        epoch_bar = tqdm(range(iterations), desc="Epoch progress", leave=False)
        for i in epoch_bar:
            # TODO build pairs and shuffle.
            bar = tqdm(range(epoch_data[i].shape[-1] - 1), desc="Time Loop", leave=False)
            for t in bar:
                opt.zero_grad()
                input_x = torch.tensor(epoch_data[i][:, :, :, t].swapaxes(0, 1))
                output_y = torch.tensor(epoch_data[i][:, :, :, t+1].swapaxes(0, 1))
                input_x_re = torch.reshape(input_x, [batch_size, -1]).unsqueeze(1)
                output_y_hat = network(input_x_re)
                output_y_hat = torch.reshape(output_y_hat.squeeze(), [batch_size, 4, 2])
                loss = loss_fun(output_y, output_y_hat)

                # energy conservation loss
                def get_kinetic_energy(velocity, mass=1.):
                    return 0.5*mass*torch.sum(velocity**2, dim=-1)

                norm = lambda v: torch.sqrt(torch.sum(v**2, dim=-1))
                def get_potential_energy(r, const_g=1, m1=1., m2=.1):
                    return -const_g*m1*m2/norm(r)

                if conserve:
                    p1_init, p2_init, v1_init, v2_init = torch.split(input_x, 1, dim=1)
                    kin_energy_init = get_kinetic_energy(v1_init) + get_kinetic_energy(v2_init) 
                    pot_energy_init = get_potential_energy(p2_init - p1_init)
                    tot_energy_init = kin_energy_init + pot_energy_init

                    p1_pred, p2_pred, v1_pred, v2_pred = torch.split(output_y_hat, 1, dim=1)
                    kin_energy_pred = get_kinetic_energy(v1_pred) + get_kinetic_energy(v2_pred) 
                    pot_energy_pred = get_potential_energy(p2_pred - p1_pred)
                    tot_energy_pred = kin_energy_pred + pot_energy_pred

                    energy_loss = loss_fun(tot_energy_init, tot_energy_pred)
                    loss = loss + energy_loss

                loss.backward()
                opt.step()
                dloss = loss.detach().numpy()
                deng = energy_loss.detach().numpy()
                bar.set_description(f"Prediction loss: {dloss:3.8f}, Energy: {deng:3.8f}")
            loss, net_out, sim_out = test_net(network, 2)
            epoch_bar.set_description(f"Epoch progress, test-loss: {float(loss.detach().numpy()):2.2f}")

            torch.save(network, f"network_c{conserve}.pt")

    print("training done")
    torch.save(network, f"network_c{conserve}.pt")

    pass