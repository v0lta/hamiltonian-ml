from sim_2_body import simulate, get_potential_energy, get_kinetic_energy


import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter

# Fixing random state for reproducibility
np.random.seed(19680801)


metadata = dict(title='Energy Movie', artist='Matplotlib',
                comment='Conservation visualization!')
writer = FFMpegWriter(fps=30, metadata=metadata)

fig, axs = plt.subplots(1, 2)
plot_p1, = axs[0].plot([], [], 'b-o')
plot_p2, = axs[0].plot([], [], 'g-o')
plot_l1, = axs[0].plot([], [], 'b')
plot_l2, = axs[0].plot([], [], 'g')

plot_ek, = axs[1].plot([], [], label='kinetic')
plot_ep, = axs[1].plot([], [], label='potential')
plot_et, = axs[1].plot([], [], label='total')

G=1.
m1=1.
m2=1.
std = 0.005
t_max = 800

axs[0].set_title("Position")
axs[0].set_xlim(-20, 20)
axs[0].set_ylim(-20, 20)

axs[1].set_title("Energy")
axs[1].set_xlim(0, t_max)
axs[1].set_ylim(-1.1, 1.1)

plt.legend()


init = [np.array([0., 0.]),
        np.array([0., .97]),
        np.array([1., 0.]),
        np.array([-1., 0.])]
p1, p2, v1, v2, t_points, _ = simulate(init, seed = - 1, std = std, G=G, m1=m1, m2=m2, t_max=t_max)
t_points = t_points[::25]
p1 = p1[:, ::25]
p2 = p2[:, ::25]
v1 = v1[:, ::25]
v2 = v2[:, ::25]

potential_energy = np.array([get_potential_energy(G, m1, m2, cp2-cp1) for cp1, cp2 in zip(p1.swapaxes(0, 1), p2.swapaxes(0, 1))])
kinetic_energy1 = np.array([get_kinetic_energy(m1, cv1) for cv1 in v1.T])
kinetic_energy2 = np.array([get_kinetic_energy(m2, cv2) for cv2 in v2.T])

kinetic_energy = kinetic_energy1 + kinetic_energy2
total_energy = potential_energy + kinetic_energy


with writer.saving(fig, "two_body_sim.mp4", 100):
    for i in range(1, len(p1[0])):
        plot_l1.set_data(p1[0, :i], p1[1, :i])
        plot_l2.set_data(p2[0, :i], p2[1, :i])
        plot_p1.set_data([p1[0, i]], [p1[1, i]])
        plot_p2.set_data([p2[0, i]], [p2[1, i]])

        plot_ek.set_data(t_points[:i], kinetic_energy[:i])
        plot_ep.set_data(t_points[:i], potential_energy[:i])
        plot_et.set_data(t_points[:i], total_energy[:i])

        writer.grab_frame()
