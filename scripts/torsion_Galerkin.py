import torch
import shutil
import os
from tqdm import tqdm
import math
from time import time 
import csv 

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_shape

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

OUTPUT_FOLDER = 'res/torsion_Galerkin'
DIM = 2
N_UNIT = 30
N_POINTS = 20_000

# Create the output folder
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

# Create the model
model = ConvexDiffeo(input_size=DIM, n_unit=N_UNIT).to(device)

# Set up optimizer
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=.1,
    line_search_fn="strong_wolfe"
)

# Source term
def f(x):
    return 0 * x[...,0] + 1

# For the l-bfgs optimizer
def closure():
    optimizer.zero_grad()

    # Compute the solution to the Poisson problem
    u_coefs, sources = model.poisson_problem(f, n_sources=600, n_quad_points=N_POINTS)
    u = lambda x: model.evaluate_rbf_function(x, sources, u_coefs)

    # Compute the torsion
    x = model.sample_ball(N_POINTS)
    det = torch.abs(torch.linalg.det(model.jacobian(x)))
    omega = math.pi ** (DIM/ 2) / math.gamma(DIM / 2 + 1)
    integrand = u(x) * det
    tor = omega * integrand.mean()

    # Compute the loss
    vol = model.volume(n_points=N_POINTS)
    loss = -tor/((vol**(DIM+2)/DIM)) + (vol-1)**2

    loss.backward()

    return loss


# Optimize
start = time()
time_history = []
deficit_history = []

for i in tqdm(range(50)):
    time_history.append(time()-start)

    with torch.no_grad():
        omega = math.pi ** (DIM / 2) / math.gamma(DIM / 2 + 1)
        c = DIM * omega**(1/DIM)
        vol = model.volume(n_points=100_000)
        per = model.perimeter(n_points=100_000)
        isoper_deficit = per / (c * vol**((DIM-1)/DIM)) - 1

    deficit_history.append(isoper_deficit.item())

    loss = optimizer.step(closure)

    print(f'Loss = {loss.item()}\t Isoperimetric deficit = {isoper_deficit.item()}')

# Put the loss in a csv file
os.makedirs('res/csvs/', exist_ok=True)

output = os.path.join('res/csvs/', f'history_Galerkin.csv')
with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "deficit"])

    for t, l in zip(time_history, deficit_history):
        writer.writerow([t, l])


