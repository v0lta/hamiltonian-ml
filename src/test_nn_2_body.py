import numpy as np
import matplotlib.pyplot as plt
from sim_2_body import simulate
from scipy.integrate import solve_ivp

import torch
import numpy as np
from networks import Feedforward, get_hamiltonian
from sim_2_body import simulate, get_kinetic_energy, get_potential_energy
from sim_2_body import absolute_motion



def test_net(network, data_mean, data_std, seed = -1, conserve=True):
    t_0 = 0.
    t_max = 100
    # dt = 0.01

    init = [np.array([2, 0.]),
            np.array([-2, 0.])]

    sim = simulate(init, seed = seed, t_max=t_max)
    p1, p2, v1, v2, t_points, _, init = sim

    skip = 1
    p1 = p1[:, ::skip]
    p2 = p2[:, ::skip]
    v1 = v1[:, ::skip]
    v2 = v2[:, ::skip]
    t_points = t_points[::skip]
    sim_out = ([p1, p2, v1, v2, t_points])

    y0 = np.expand_dims(np.stack(init), 0)


    if conserve:
        def modelwrap(_, y):
            y = np.reshape(y, [1, 4, 2])
            y = torch.tensor(y.astype(np.float32))
            network.zero_grad()
            ydot = get_hamiltonian(y, out_y=None, network=network)[1]
            ydot = ydot.detach().numpy()
            return ydot.flatten()
    else:
        def modelwrap(_, y):
            y = np.reshape(y, [1, 4, 2])
            y = torch.tensor(y.astype(np.float32))
            with torch.no_grad():
                ydot = network(y)
            ydot = ydot.detach().numpy()
            return ydot.flatten()        


    sol = solve_ivp(modelwrap, [t_0, t_max], y0.flatten(), t_eval=t_points)
    p1, p2, v1, v2 = np.split(sol.y, 4)
    net_out = (p1, p2, v1, v2)
    
    net_out_inv = net_out

    # data_mean = data_mean.squeeze(2).squeeze(-1)
    # data_std = data_std.squeeze(2).squeeze(-1)
    # y0 = (y0 - data_mean)/data_std

    #input_x = torch.tensor(y0.astype(np.float32))
    #output_y_net = [input_x]
    #for _ in range(t_points.shape[-1]-1):
    #    if not conserve:
    #        dx_dt = network(output_y_net[-1])
    #    else:
    #        dx_dt = get_hamiltonian(output_y_net[-1], out_y=None, network=network)[1]
    #    out = output_y_net[-1] + dt*dx_dt
    #    output_y_net.append(out)
    #output_y_net = [torch.squeeze(y_el, 0) for y_el in output_y_net]
    #net_out = torch.stack(output_y_net, -1).detach().numpy()
    # net_out_inv = (net_out*data_std.transpose([1,2,0])) + data_mean.transpose([1,2,0])

    loss = np.mean(list(np.mean((net - sim)**2) for net, sim in zip(net_out_inv, sim_out[:4])))
    return loss, net_out, sim_out


if __name__ == '__main__':
    seed = 150
    conserve = True
    print(f"conserve: {conserve}")
    network, mean, std = torch.load(f"network_c{conserve}.pt")
    loss, net_out, sim_out = test_net(network, data_mean=mean, data_std=std, seed=seed)
    print(f"loss: {float(loss):2.2f}")
    net_out_numpy = net_out
    net_p1, net_p2, net_v1, net_v2 = (net_out_numpy[i] for i in range(4))
    p1, p2, v1, v2, t_points = sim_out

    print(f"net_out_shape {net_p1.shape}")

    # plot position
    plt.plot(p1[0, :], p1[1, :], 'b')
    plt.plot(p1[0, 0], p1[1, 0], 'b.')
    plt.plot(p2[0, :], p2[1, :], 'g')
    plt.plot(p2[0, 0], p2[1, 0], 'g.')

    plt.plot(net_p1[0, :], net_p1[1, :], 'c')
    plt.plot(net_p1[0, 0], net_p1[1, 0], '.c')
    plt.plot(net_p2[0, :], net_p2[1, :], 'm')
    plt.plot(net_p2[0, 0], net_p2[1, 0], '.m')
    plt.show()

    # plot energy over time
    potential_energy = np.array([get_potential_energy(1., 1., 1., cp2-cp1) 
                                 for cp1, cp2 in zip(net_p1.swapaxes(0, 1), net_p2.swapaxes(0, 1))])
    kinetic_energy1 = np.array([get_kinetic_energy(1., cv1) for cv1 in net_v1.T])
    kinetic_energy2 = np.array([get_kinetic_energy(1., cv2) for cv2 in net_v2.T])
    
    kinetic_energy = kinetic_energy1 + kinetic_energy2
    total_energy = potential_energy + kinetic_energy

    plt.title("Energy")
    plt.plot(t_points, potential_energy, label='potential')
    plt.plot(t_points, kinetic_energy, label='kinetic')
    plt.plot(t_points, total_energy, label='total')
    # plt.ylim([-1.1, 1.1])
    plt.legend()
    plt.show()
    pass