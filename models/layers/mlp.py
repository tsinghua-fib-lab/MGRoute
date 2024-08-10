import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(mlp, self).__init__()
        self.layers = nn.ModuleList()
        len_layer_size = len(layer_sizes)
        for i in range(len_layer_size):
            if i == 0:
                self.layers.append(nn.Linear(input_size, layer_sizes[i]))
            else:
                self.layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
    
class mlp_res(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim = 0):
        super(mlp_res, self).__init__()
        if out_dim == 0:
            self.fc1 = mlp(in_dim, [hidden_dim, hidden_dim])
        else:
            self.fc1 = mlp(in_dim, [hidden_dim, out_dim])
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        out = self.fc1(x)
        # dropout
        out = self.dropout(out)
        return x + out
        # return x + self.fc1(x)

class QuantileMLP(nn.Module):
    def __init__(self, input_dim, quantile_size=9, name=''):
        super(QuantileMLP, self).__init__()
        self.quantile_size = quantile_size
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for _ in range(quantile_size)
        ])

    def forward(self, inputs):
        output_list = []
        for mlp in self.mlps:
            quantile_output = mlp(inputs)
            output_list.append(quantile_output)
        return torch.cat(output_list, dim=-1)