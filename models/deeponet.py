import torch
import torch.nn as nn

class DeepONet(nn.Module):
    def __init__(self, branch_dim: int, trunk_dim: int, hidden_dim: int = 128, p: int = 100):
        super().__init__()
        # Branch Network (5-Hidden Layer MLP)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, p)
        )
        # Trunk Network (5-Hidden Layer MLP)
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, p)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = self.branch_net(u)
        T = self.trunk_net(y)
        out = torch.sum(B * T, dim=1, keepdim=True) + self.bias
        return out