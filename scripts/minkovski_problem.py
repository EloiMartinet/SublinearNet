import torch
import shutil
import os
from tqdm import tqdm

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_shape

# Important for curvature computation
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# Parameters
DIM = 3
N_QUAD = 1_000
OUTPUT_FOLDER = f'res/minkovski_problem'

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

# Create the output foler
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create the model
model = ConvexDiffeo(
    input_size=DIM,
    n_unit=500,
    mode='support'
).to(device)

# Set up optimizer
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=.02,
    line_search_fn="strong_wolfe"
)

def positive_function(x):
    # return 0 * x[:,0] + 1 
    return 0.25*9*(torch.sin(6*x[:,0]) + 1.0)*(torch.cos(6*x[:,1]) + 1.0) + 0.5

# g must verify \int_S x/g(x) dS(x)=0; we take g to be symmetric which ensures it
def g(x):
    return positive_function(x) + positive_function(-x)

def error(x):
    curvature = model.gaussian_curvature(x)
    target_curvature = g(x)
    rel_diff = (target_curvature-curvature)/target_curvature

    return rel_diff**2

def log_rel_error(x):
    curvature = model.gaussian_curvature(x)
    target_curvature = g(x)
    rel_err = torch.abs(curvature-target_curvature)/target_curvature

    return torch.log10(rel_err)

# For the l-bfgs optimizer
def closure():
    optimizer.zero_grad()
    x = model.sample_sphere(n_points=N_QUAD)
    loss = torch.mean(error(x))

    print(f'{loss.item()=}')
    loss.backward()

    return loss

# Optimize
for step in tqdm(range(1_000)):
    with torch.no_grad():
        plot_shape(
            model, 
            n_points=150,
            color_fn = log_rel_error, 
            output = os.path.join(OUTPUT_FOLDER, f'{step}.png')
        )
        plot_shape(
            model, 
            n_points=150,
            color_fn = g, 
            output = os.path.join(OUTPUT_FOLDER, f'target_{step}.png')
        )
    loss = optimizer.step(closure)
    print(f'Loss = {loss.item()}')
