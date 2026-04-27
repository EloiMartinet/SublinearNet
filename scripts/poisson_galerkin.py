import torch
import shutil
import os
from tqdm import tqdm
import math

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_shape


torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

OUTPUT_FOLDER = 'res/poisson_galerkin'
DIM = 2
N_UNIT = 20
N_POINTS = 20_000

def f1(x, c=torch.tensor([0.3, 0.0])):
    """
    x: tensor of shape (batch_size, 2)
    returns: tensor of shape (batch_size,)
    """
    c1 = c[0]
    c2 = c[1]
    x1 = x[..., 0]
    x2 = x[..., 1]
    
    return 20 * ((x1-c1) + 0.4 - (x2-c2)**2)**2 + (x1-c1)**2 + (x2-c2)**2 - 1

def f2(x, n=5):
    """
    x: tensor of shape (batch_size, 2)
    returns: tensor of shape (batch_size,)
    """
    x1 = x[:, 0:1]  # (batch, 1)
    x2 = x[:, 1:2]  # (batch, 1)

    i = torch.arange(n, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, n)

    # Constants
    two_pi_over_n = 2 * torch.pi / n

    # y_i
    y1 = torch.sin((i + 0.5) * two_pi_over_n)  # (1, n)
    y2 = torch.cos((i + 0.5) * two_pi_over_n)  # (1, n)

    # z_i
    z1 = (6.0 / 5.0) * torch.sin(i * two_pi_over_n)  # (1, n)
    z2 = (6.0 / 5.0) * torch.cos(i * two_pi_over_n)  # (1, n)

    # Exponentials
    term1 = torch.exp(-8 * ((x1 - y1)**2 + (x2 - y2)**2))  # (batch, n)
    term2 = torch.exp(-8 * ((x1 - z1)**2 + (x2 - z2)**2))  # (batch, n)

    sum1 = term1.sum(dim=1)
    sum2 = term2.sum(dim=1)
    
    x1 = x1.squeeze()
    x2 = x2.squeeze()

    return -0.5 + (4.0 / 5.0) * (x1**2 + x2**2) + 2 * sum1 - sum2

# Pick the source term 
f = f2  

# Create the output folder
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create the model
model = ConvexDiffeo(input_size=DIM, n_unit=N_UNIT)

# Set up optimizer
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=.1,
    line_search_fn="strong_wolfe"
)


# For the l-bfgs optimizer
def closure():
    optimizer.zero_grad()

    # Compute the solution to the Poisson problem
    u_coefs, sources = model.poisson_problem(f, n_sources=1000, n_quad_points=N_POINTS)
    u = lambda x: model.evaluate_rbf_function(x, sources, u_coefs)

    # Compute the integral
    x = model.sample_ball(N_POINTS)
    det = torch.abs(torch.linalg.det(model.jacobian(x)))
    omega = math.pi ** (DIM/ 2) / math.gamma(DIM / 2 + 1)
    integrand = u(x) * det
    loss = (omega * integrand.mean()).unsqueeze(0)

    loss.backward()

    return loss

# Optimize
for i in tqdm(range(1_000)):
    with torch.no_grad():
        u_coefs, sources = model.poisson_problem(f)
        u = lambda x: model.evaluate_rbf_function(x, sources, u_coefs)

        output = os.path.join(OUTPUT_FOLDER, f'{i}.png')
        plot_shape(
            model, 
            output=output,
             n_points=200, 
             background_fn=f 
        )
        
    loss = optimizer.step(closure)
    print(f'Loss = {loss.item()}')


