import torch

class Feedforward(torch.nn.Module):
    '''A standard MLP'''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        self.activation = torch.nn.Tanh()

    def forward(self, x, dt=0.5):
        return x + dt*self.dxdt(x)


    def dxdt(self, x):
        batch_size, vars, dims = x.shape
        x_re = torch.reshape(x, [batch_size, self.input_dim]).unsqueeze(1)
        h = self.activation(self.linear1(x_re))
        h = self.activation(self.linear2(h))
        h = self.linear3(h)
        h = torch.reshape(h, [batch_size, vars, dims])
        return h