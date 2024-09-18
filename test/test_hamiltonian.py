import torch


import sys
sys.path.insert(0, "./src/")

from networks import Feedforward
from train_nn_2_body import get_hamiltonian
from sim_2_body import absolute_motion


class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        # assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1, 2)

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        out = self.forward(x) # traditional forward pass
        F = out[0]

        conservative_field = torch.zeros_like(x) # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'conservative':
            dF = torch.autograd.grad(F.sum(), x, create_graph=True)[0] # gradients for solenoidal field
            solenoidal_field = dF @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M



def test_hamiltonian():
    network = torch.load("network_cTrue.pt")[0]

    hnn = HNN(2*4, network)

    x = torch.ones([3, 4, 2])
    x.requires_grad_()


    xdot = hnn.time_derivative(x.reshape(3, 1, 8))
    xdot = torch.reshape(xdot, [3, 4, 2])
    # motion_wrapper = lambda x: absolute_motion(None, x)
    xdot2 = get_hamiltonian(x, out_y=None, network=network)[1]
    assert torch.allclose(xdot, xdot2)
    print('stop')
    pass