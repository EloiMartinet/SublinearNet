import torch
from invertible_nn import ConvexDiffeo
from plot_utils import plot_shape
import shutil
import os
from tqdm import tqdm
import math
from PINN import DirichletPINN
from time import time
import csv
import itertools
from datetime import datetime


# =========================
# Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT = f'res/hparam_search_{timestamp}'
DIM = 2
N_UNITS_DIFFEO = 30
N_POINTS = 20_000

# Search parameters
N_UNITS = [32, 64]
ACTIVATIONS = ['tanh', 'sin']
PINN_DEPTHS = [4, 3, 2]
OPTIMIZERS = ['Adam', 'LBFGS']
LEARNING_RATES = [1e-1, 1e-2, 1e-3]
MAX_TIME = 1800 # Allows 0 minutes of maximal run time


shutil.rmtree(BASE_OUTPUT, ignore_errors=True)
os.makedirs(BASE_OUTPUT, exist_ok=True)

def create_optimizer(opt_name, params, lr):
    if opt_name == 'LBFGS':
        return torch.optim.LBFGS(
            params,
            lr=lr,
            line_search_fn="strong_wolfe"
        )
    elif opt_name == 'Adam':
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer {opt_name}")

# Grid search
for n_units, activation_fn, depth, opt_name, lr in itertools.product(N_UNITS, ACTIVATIONS, PINN_DEPTHS, OPTIMIZERS, LEARNING_RATES):

    print(f"\n=== Running config: units={n_units}, activation={activation_fn}, depth={depth}, opt={opt_name}, lr={lr} ===")

    OUTPUT_FOLDER = os.path.join(BASE_OUTPUT, f'n{n_units}_{activation_fn}_d{depth}_{opt_name}_lr{lr}')
    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # =========================
    # Create model + move to device
    # =========================
    diffeo = ConvexDiffeo(input_size=DIM, n_unit=N_UNITS_DIFFEO).to(device)
    pinn = DirichletPINN(DIM, diffeo, depth=depth, hidden_dim=n_units, activation=activation_fn).to(device)

    optimizer = create_optimizer(opt_name, pinn.parameters(), lr)

    # Closure for optimizer
    def closure():
        optimizer.zero_grad()
        dir_energy = pinn.dirichlet_energy(n_quad_points=N_POINTS)
        vol = diffeo.volume(n_points=N_POINTS)
        loss = dir_energy / ((vol**(DIM+2)/DIM)) + (vol-1)**2
        loss.backward()
        return loss

    start = time()
    time_history = []
    deficit_history = []

    for i in tqdm(range(10000)):

        with torch.no_grad():
            omega = math.pi ** (DIM / 2) / math.gamma(DIM / 2 + 1)
            c = DIM * omega**(1/DIM)

            vol = diffeo.volume(n_points=50_000)
            per = diffeo.perimeter(n_points=50_000)

            isoper_deficit = per / (c * vol**((DIM-1)/DIM)) - 1
        

        loss = optimizer.step(closure)

        print(f'Loss = {loss.item()}\t Isoperimetric deficit = {isoper_deficit.item()}')

        deficit_history.append(isoper_deficit.item())
        time_history.append(time() - start)

        if time() - start > MAX_TIME:
            break

    # =========================
    # Save CSV (move to CPU safely)
    # =========================
    csv_folder = os.path.join(BASE_OUTPUT, 'csvs')
    os.makedirs(csv_folder, exist_ok=True)

    csv_path = os.path.join(csv_folder, f'n{n_units}_{activation_fn}_d{depth}_{opt_name}_lr{lr}.csv')
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "deficit"])
        for t, d in zip(time_history, deficit_history):
            writer.writerow([t, float(d)])  # ensure CPU scalar