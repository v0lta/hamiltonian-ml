import torch

class Feedforward(torch.nn.Module):
    '''A standard MLP'''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        self.activation = torch.nn.Tanh()

    def forward(self, x):
        h = self.activation(self.linear1(x))
        h = self.activation(self.linear2(h))
        return x + self.linear3(h)


    def dxdt(self, x):
        h = self.activation(self.linear1(x))
        h = self.activation(self.linear2(h))
        return self.linear3(h)