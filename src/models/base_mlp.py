import torch


class BaseMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, n_classes=0):
        super().__init__()
        self.net = []
        self.net.append(torch.nn.Flatten())
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(activation())
        for _ in range(num_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim // 2))
            self.net.append(activation())
            hidden_dim //= 2
        self.net.append(torch.nn.Linear(hidden_dim, n_classes))
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)