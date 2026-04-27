import torch
import torch.nn as nn
import torch.autograd as autograd
import math

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class DirichletPINN(nn.Module):
    def __init__(self, dim, diffeo, hidden_dim=64, depth=3, activation='sin'):
        super().__init__()
        layers = []
        dims = [dim] + [hidden_dim]*depth + [1]
        
        if activation == 'sin':
            self.activation = Sin()
        if activation == 'tanh':
            self.activation = nn.Tanh()

        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(self.activation)
        
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)
        self.dim = dim
        self.diffeo = diffeo

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=1, keepdim=True)
        res = (1.0 - norm) * self.mlp(x)

        return res

    # Energy functional
    def dirichlet_energy(self, n_quad_points=10_000):
        """
        x: (N, d)
        A_fn: function mapping (N, d) -> (N, d, d)
        rho_fn: function mapping (N, d) -> (N, 1)
        """
        x = self.diffeo.sample_ball(n_points=n_quad_points)
        x.requires_grad_(True)
        
        v = self.forward(x)              
        grad_v = autograd.grad(
            outputs=v,
            inputs=x,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Set the change of variable matrix and weight
        jac = self.diffeo.jacobian(x)
        rho = torch.linalg.det(jac)
        inv_jac = torch.linalg.inv(jac)
        A = rho[:, None, None] * (
            inv_jac @ inv_jac.transpose(1, 2)
        )

        # compute A ∇v
        A_grad_v = torch.bmm(A, grad_v.unsqueeze(-1)).squeeze(-1)

        # energy density
        energy_density = 0.5 * torch.sum(grad_v * A_grad_v, dim=1, keepdim=True) \
                        - rho * v

        omega = math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1)  # Volume of the unit ball

        return omega * energy_density.mean()