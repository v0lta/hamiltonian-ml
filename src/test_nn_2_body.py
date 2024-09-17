import numpy as np
import matplotlib.pyplot as plt
from sim_2_body import simulate
from scipy.integrate import solve_ivp

import torch
import numpy as np
from networks import Feedforward
from sim_2_body import simulate, get_kinetic_energy, get_potential_energy




def test_net(network, seed = -1):
    G=1.
    m1=1.
    m2=1.
    t_0 = 0.
    std = 0.0025
    t_max = 600
    dt = 0.5

    init = [np.array([0., 0.]),
            np.array([0., .97]),
            np.array([1., 0.]),
            np.array([-1., 0.])]

    sim = simulate(init, seed = seed, std = std, G=G, m1=m1, m2=m2, t_max=t_max)
    p1, p2, v1, v2, t_points, _ = sim

    skip = 1
    p1 = p1[:, ::skip]
    p2 = p2[:, ::skip]
    v1 = v1[:, ::skip]
    v2 = v2[:, ::skip]
    t_points = t_points[::skip]
    sim_out = ([p1, p2, v1, v2, t_points])

    y0 = np.expand_dims(np.stack(init), 0)

    # def modelwrap(_, y):
    #     y = np.reshape(y, y0.shape)
    #     y = torch.tensor(y.astype(np.float32))
    #     with torch.no_grad():
    #         ydot = network.dxdt(y)
    #     ydot = ydot.numpy()
    #     return ydot.flatten()


    # sol = solve_ivp(modelwrap, [t_0, t_max], y0.flatten(), t_eval=t_points)
    # p1, p2, v1, v2 = np.split(sol.y, 4)
    # net_out = (p1, p2, v1, v2)
    

    input_x = torch.tensor(y0.astype(np.float32))
    output_y_net = [input_x]
    for _ in range(t_points.shape[-1]-1):
        out = output_y_net[-1] + dt*network.dxdt(output_y_net[-1])
        output_y_net.append(out)
    output_y_net = [torch.squeeze(y_el, 0) for y_el in output_y_net]
    net_out = torch.stack(output_y_net, -1).detach().numpy()
    
    loss = np.mean(list(np.mean((net - sim)**2) for net, sim in zip(net_out, sim_out)))
    return loss, net_out, sim_out


if __name__ == '__main__':
    seed = 1
    network = torch.load("network_cFalse.pt")
    loss, net_out, sim_out = test_net(network, seed)
    print(f"loss: {float(loss):2.2f}")
    net_out_numpy = net_out
    net_p1, net_p2, net_v1, net_v2 = (net_out_numpy[i] for i in range(4))
    p1, p2, v1, v2, t_points = sim_out

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