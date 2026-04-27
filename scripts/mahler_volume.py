import torch
import shutil
import os
import math
from functools import partial
from tqdm import tqdm

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_shape

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# Parameters
DIM = 2
N_SYMMETRIES = 3          # <-- number of rotational symmetries (n-fold)
N_QUAD = 100_000
OUTPUT_FOLDER = f'res/mahler_volume'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

# Rotations
def rotation_matrix(theta):
    """2D rotation matrix."""
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor([[c, -s],
                         [s,  c]], device=device)


def rotate(x, R):
    """
    Apply rotation to a batch of points.
    x: (..., 2)
    R: (2, 2)
    """
    return x @ R.T


# Build symmetry group: cyclic rotations by 2πk/N
symmetries = []
for k in range(N_SYMMETRIES):
    theta = 2 * math.pi * k / N_SYMMETRIES
    Rk = rotation_matrix(theta)
    symmetries.append(partial(rotate, R=Rk))

# Output folder
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Model with symmetry group
model = ConvexDiffeo(
    input_size=DIM,
    n_unit=30,
    symmetries=symmetries,
    mode='support'
).to(device)

# Set up optimizer
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=.02,
    line_search_fn="strong_wolfe",
)

# Optimal Mahler volume (see https://arxiv.org/pdf/1507.01481)
optimal_value = N_SYMMETRIES**2 * math.sin(math.pi/N_SYMMETRIES)**2

# For the l-bfgs optimizer
def closure():
    optimizer.zero_grad()
    mahler_vol = model.mahler_volume(n_points=N_QUAD)
    vol = model.volume(n_points=N_QUAD)

    # L2 regularization
    l2_reg = 0.0
    for p in model.parameters():
        l2_reg += (p**2).sum()

    loss = mahler_vol + (vol-1)**2 + 2e-6*l2_reg

    loss.backward()

    return loss

# Optimize
for step in tqdm(range(1_000)):
    # Plot
    output = os.path.join(OUTPUT_FOLDER, f'{step}.png')
    plot_shape(
        model, output=output, n_points=2000
    )

    # Compute loss
    loss = optimizer.step(closure)

    # Print
    with torch.no_grad():
        mahler_vol = model.mahler_volume(n_points=N_QUAD)

    print(f'Loss = {loss.item()}\t Mahler volume = {mahler_vol.item()}\t Optimal value = {optimal_value}')
