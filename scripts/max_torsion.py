import torch
import shutil
import os
from tqdm import tqdm

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_shape


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

OUTPUT_FOLDER = 'res/max_torsion'
DIM = 3
N_UNIT = 200
PROBLEM = "volume"
N_POINTS = 100_000

# Create the output folder
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create the model
model = ConvexDiffeo(input_size=DIM, n_unit=N_UNIT)

# Set up optimizer
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=.05,
    line_search_fn="strong_wolfe"
)

# For the l-bfgs optimizer
def closure():
    optimizer.zero_grad()

    max_tor = model.grad_norm_infinity_boundary(tol=1e-5)

    if PROBLEM == 'volume':
        vol = model.volume(n_points=N_POINTS)
        loss = -max_tor/vol**(1/DIM) + (vol-1)**2 # Adding a volume penalization to avoid degeneracy
        print(f'Functional value ={(max_tor/vol**(1/DIM)).item()}')

    if PROBLEM == 'perimeter':
        per = model.perimeter(n_points=N_POINTS)
        loss = -max_tor/per**(1/(DIM-1)) + (per-1)**2 # Adding a perimeter penalization to avoid degeneracy
        print(f'Functional value ={(max_tor/per**(1/(DIM-1))).item()}')
    
    loss.backward()

    return loss

# Optimize
for i in tqdm(range(1_000)):
    output = os.path.join(OUTPUT_FOLDER, f'{i}.png')
    plot_shape(model, output=output)#, video=True)
    loss = optimizer.step(closure)
    print(f'Loss = {loss.item()}')


