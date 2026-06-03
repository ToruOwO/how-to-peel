import torch
import torch.nn as nn


class ResidualHead(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, n_layers=2):
        super().__init__()
        layers = []
        dims = [obs_dim] + [hidden]*n_layers + [act_dim]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        
    def forward(self, obs):
        return self.net(obs)
    

class RewardModel(nn.Module):
    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int,
                 hidden_dim: int = 256, 
                 num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = obs_dim + act_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(
        self, 
        z: torch.Tensor, 
        a: torch.Tensor, 
        return_feat: bool = False
    ) -> torch.Tensor:
        x = torch.cat([z, a], dim=-1)   # [B, obs_dim + act_dim]
        h = x
        for layer in self.net[:-1]:
            h = layer(h)
        r = self.net[-1](h).squeeze(-1) # [B]

        if return_feat:
            return r, h                 # h: [B, hidden_dim]
        return r
    

class RewardFeatProjector(nn.Module):
    def __init__(self, in_dim: int = 256, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, in_dim]
        return self.net(feat)           # [B, out_dim]
