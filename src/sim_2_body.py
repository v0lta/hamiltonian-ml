import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

norm = lambda v: np.sqrt(np.sum(v**2))

def get_grav_force(G, m1, m2, r):
    return G * (m1* m2)/np.sum(r**2)

def get_kinetic_energy(mass, velocity):
    return 0.5*mass*np.sum(velocity**2)

def get_potential_energy(const_g, m1, m2, r):
    return -const_g*m1*m2/norm(r)


def absolute_motion(_, y, G=1., m1=1., m2=1.):
    """Compute motion from the forces of gravity.
    
    Fg = G (m1 m2)/(r**2)
    F = ma
    a = F/m
    See also: https://orbital-mechanics.space/the-n-body-problem/two-body-inertial-numerical-solution.html
    """
    p1, p2, v1, v2 = np.split(y, 4)
    dist_r = p2 - p1

    # get gravitational forces
    grav = get_grav_force(G, m1, m2, dist_r)
    # add direction
    fg1 = grav * dist_r / norm(dist_r)
    fg2 = - fg1
    
    a1 = fg1/m1
    a2 = fg2/m2
    # pos1, pos2, vel1, vel2 turn into their integrated counterparts.
    ydot = np.concatenate([v1, v2, a1, a2])
    return ydot


def simulate(init, seed = -1, std = 0.05, t_max=50., dt = 0.5):
    #     p1, p2, v1, v2 = init
    # shift the position of the ring.
    p1 = init[0]
    p2 = init[1]
    dist = p2 - p1
    if seed >= 0:
        rng = np.random.default_rng(seed)
        p1_new = p1 + rng.normal(scale=std, size=[2])
    else:
        p1_new = p1
    p2_new = p1_new + dist
    v1, v2 = get_v_circ(p1, p2)
    init = [p1_new, p2_new, v1, v2]
        
    t_0 = 0.
    steps = int(t_max/dt)
    t_points = np.linspace(t_0, t_max, steps)
    y0 = np.concatenate(init)

    sol = solve_ivp(absolute_motion, [t_0, t_max], y0, t_eval=t_points)
    p1, p2, v1, v2 = np.split(sol.y, 4)
    return p1, p2, v1, v2, t_points, sol.success, y0


def get_v_circ(p1, p2):
    d = p2 - p1
    r = np.sqrt( np.sum(d **2))
    v = np.sqrt( 1/r ) / 2.

    v1 = v*np.flipud(d)/norm(d)
    v2 = -v*np.flipud(d)/norm(d)
    return v1, v2

if __name__ == '__main__':
    G=1.
    m1=1.
    m2=1.
    t_max = 50

    init = [np.array([2, 0.]),
            np.array([-2, 0.])]

    res = [simulate(init, seed = seed - 1, t_max=t_max) for seed in range(10)]
    # res = list(filter(lambda r: r[-1] == True, res))
    print(f"{len(res)}")

    for r in res:
        p1, p2, v1, v2, t_points, _, _ = r

        print(f"mp1 {np.mean(p1)}, mp2 {np.mean(p2)}")

        # plot position
        plt.title("Position")
        plt.plot(p1[0, :], p1[1, :])
        plt.plot(p2[0, :], p2[1, :])
    plt.show()


    for r in res:
        p1, p2, v1, v2, t_points, _, _ = r

            # plot position
        plt.title("Position")
        plt.plot(p1[0, :], p1[1, :])
        plt.plot(p2[0, :], p2[1, :])
        plt.show()

        # plot energy over time
        potential_energy = np.array([get_potential_energy(G, m1, m2, cp2-cp1) for cp1, cp2 in zip(p1.swapaxes(0, 1), p2.swapaxes(0, 1))])
        kinetic_energy1 = np.array([get_kinetic_energy(m1, cv1) for cv1 in v1.T])
        kinetic_energy2 = np.array([get_kinetic_energy(m2, cv2) for cv2 in v2.T])
        
        kinetic_energy = kinetic_energy1 + kinetic_energy2
        total_energy = potential_energy + kinetic_energy

        plt.title("Energy")
        plt.plot(t_points, potential_energy, label='potential')
        plt.plot(t_points, kinetic_energy, label='kinetic')
        plt.plot(t_points, total_energy, label='total')
        plt.legend()
        plt.show()

        # y = np.concatenate([p1, p2, v1, v2])
        # ydot = absolute_motion(_, y)
        # ydot_est = np.stack([(y[:, t] - y[:, t-1])/0.5 for t in range(1, y.shape[-1])], -1)
        # plt.semilogy(np.mean(np.abs(ydot), axis=0))
        # plt.semilogy(np.mean(np.abs(ydot_est), axis=0))
        # plt.semilogy(np.mean(np.abs(ydot[:, 1:] - ydot_est), axis=0))
        # plt.show()
        pass

    pass