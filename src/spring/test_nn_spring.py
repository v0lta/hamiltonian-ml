
import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)


from spring_sim import get_dataset
from networks import Feedforward


if __name__ == '__main__':
    DPI = 300
    FORMAT = 'pdf'
    LINE_SEGMENTS = 10
    ARROW_SCALE = 30
    ARROW_WIDTH = 6e-3
    LINE_WIDTH = 2
    RK4 = ''
    EXPERIMENT_DIR = './experiment-spring'

    def get_args():
        return {'input_dim': 2,
            'hidden_dim': 200,
            'learn_rate': 1e-3,
            'nonlinearity': 'tanh',
            'total_steps': 2000,
            'field_type': 'solenoidal',
            'print_every': 200,
            'name': 'spring',
            'gridsize': 10,
            'input_noise': 0.5,
            'seed': 0,
            'save_dir': './{}'.format(EXPERIMENT_DIR),
            'fig_dir': './figures'}

    class ObjectView(object):
        def __init__(self, d): self.__dict__ = d

    args = ObjectView(get_args())
    # np.random.seed(args.seed)
    field = get_field(gridsize=15)
    data = get_dataset()

    # plot config
    fig = plt.figure(figsize=(3, 3), facecolor='white', dpi=DPI)

    x, y, dx, dy, t = get_trajectory(radius=0.7, y0=np.array([1,0]))
    plt.scatter(x,y,c=t,s=14, label='data')
    plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
            cmap='gray_r', color=(.5,.5,.5))
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$\\frac{dx}{dt}$", rotation=0, fontsize=14)
    plt.title("Dynamics")
    plt.legend(loc='upper right')

    plt.tight_layout() ; plt.show()
    # fig.savefig(fig_dir + '/spring-task.png')
