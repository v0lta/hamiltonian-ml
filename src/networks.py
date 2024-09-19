import torch
from sim_2_body import absolute_motion

class Feedforward(torch.nn.Module):
    '''A standard MLP'''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        self.activation = torch.nn.Tanh()

        # for l in [self.linear1, self.linear2, self.linear4]:
        #     torch.nn.init.orthogonal_(l.weight) # use a principled initialization


    def forward(self, x):
        batch_size, _, _ = x.shape
        x_re = torch.reshape(x, [batch_size, self.input_dim]).unsqueeze(1)
        h = self.activation(self.linear1(x_re))
        h = self.activation(self.linear2(h))
        # h = self.activation(self.linear3(h))
        h = self.linear4(h)
        return h
    


def get_hamiltonian(in_x, out_y, network):
    in_x.requires_grad_()
    net_out = network(in_x)

    # p1, p2, v1, v2
    # dxdt = (output_y - input_x)/0.5
    dx_dt = torch.tensor(absolute_motion(None, in_x.detach().cpu().numpy().swapaxes(0, 1))).to(in_x.device).swapdims(0, 1)
    # if out_y is not None:
    #     dx_dt = out_y - in_x
    # else:
    #     dx_dt = None
    
    dh_dx = torch.autograd.grad(net_out.sum(), in_x, create_graph=True)[0]
    dh_dp, dh_dq = torch.split(dh_dx, 2, dim=1)
    # official code
    # dx_dt_hat = torch.cat([dh_dq, -dh_dp], dim=1)
    # paper
    dx_dt_hat = torch.cat([-dh_dq, dh_dp], dim=1)


    if out_y is not None:
        loss_fun = torch.nn.MSELoss()
        # loss_fun = torch.nn.HuberLoss()
        conservation_loss = loss_fun(dx_dt, dx_dt_hat)
    else:
        conservation_loss = None
    return conservation_loss, dx_dt_hat, dx_dt
