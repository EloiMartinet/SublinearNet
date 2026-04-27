import torch
import shutil
import os
from tqdm import tqdm
import math
from time import time
import csv

from shapes.pinn import DirichletPINN
from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_shape

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

OUTPUT_FOLDER = 'res/torsion_PINN'
DIM = 2
N_UNIT = 30
N_POINTS = 20_000
CONFIG = 'config_2'
MAX_TIME = 3600 # Run for max 1 hour

# Create the output folder
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

# Create the model
diffeo = ConvexDiffeo(input_size=DIM, n_unit=N_UNIT).to(device)

# Set up optimizer
if CONFIG == 'config_1':
    pinn = DirichletPINN(
        DIM, diffeo, activation='tanh', depth=3, hidden_dim=32
    ).to(device)
    optimizer = torch.optim.LBFGS(
        pinn.parameters(),  # The parameters of the diffeo are already part of the parameters of the PINN
        lr=.01,
        line_search_fn="strong_wolfe"
    )

if CONFIG == 'config_2':
    pinn = DirichletPINN(
        DIM, diffeo, activation='tanh', depth=3, hidden_dim=32
    ).to(device)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=.01)

if CONFIG == 'config_3':
    pinn = DirichletPINN(
        DIM, diffeo, activation='tanh', depth=2, hidden_dim=32
    ).to(device)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=.01)

# For the l-bfgs optimizer
def closure():
    optimizer.zero_grad()
    dir_energy = pinn.dirichlet_energy(n_quad_points=N_POINTS)
    vol = diffeo.volume(n_points=N_POINTS)
    loss = dir_energy/((vol**(DIM+2)/DIM)) + (vol-1)**2
    loss.backward()

    return loss

# Optimize
start = time()
time_history = []
deficit_history = []

for i in tqdm(range(10000)):
    time_history.append(time()-start)

    with torch.no_grad():
        omega = math.pi ** (DIM / 2) / math.gamma(DIM / 2 + 1)
        c = DIM * omega**(1/DIM)
        vol = diffeo.volume(n_points=100_000)
        per = diffeo.perimeter(n_points=100_000)
        isoper_deficit = per / (c * vol**((DIM-1)/DIM)) - 1

    deficit_history.append(isoper_deficit.item())

    loss = optimizer.step(closure)
    
    print(f'Loss = {loss.item()}\t Isoperimetric deficit = {isoper_deficit.item()}')


    if time() - start > MAX_TIME:
        break

# Put the loss in a csv file
os.makedirs('res/csvs/', exist_ok=True)

output = os.path.join('res/csvs/', f'history_PINN_{CONFIG}.csv')
with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "deficit"])

    for t, l in zip(time_history, deficit_history):
        writer.writerow([t, l])


