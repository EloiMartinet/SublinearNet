import torch
import shutil
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from shapes.invertible_nn import ConvexDiffeo
from shapes.plot_utils import plot_shape, plot_point_cloud_3d


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')

# Parameters
DIM = 3
N_SAMPLES = 1_000
SHAPE = 'cube'
NOISE_LEVEL = 0.02

# Set the output folder
OUTPUT_FOLDER = f'res/fit_noisy_single'

# Create the output foler
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create the target shape
shape = ConvexDiffeo(input_size=DIM, gauge_function=SHAPE).to(device)
plot_shape(
    shape, 
    n_points=3000, 
    output = os.path.join(OUTPUT_FOLDER, f'{SHAPE}.png')
)

# Plot the shape
plot_shape(
    shape, 
    n_points=1000, 
    output=os.path.join(OUTPUT_FOLDER, f'{SHAPE}.png')
)


# Create the model
model = ConvexDiffeo(
    input_size=DIM,
    n_unit=100,
    mode='gauge'
).to(device)

# Sample the target shape boundary
x = shape.sample_sphere(n_points=N_SAMPLES, random=True)
noisy_samples = shape(x)
noisy_samples += NOISE_LEVEL * torch.randn(noisy_samples.shape)

# Plot the point cloud
plot_point_cloud_3d(noisy_samples, output = os.path.join(OUTPUT_FOLDER, f'point_cloud.png'))

# Set up optimizer
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=0.1,
    max_iter=20,
    line_search_fn="strong_wolfe"
)

# For the l-bfgs optimizer
def closure():
    optimizer.zero_grad()
    y_pred = model.sublinear_nn(noisy_samples)
    loss = torch.mean((y_pred - 1)**2 )
    loss.backward()

    return loss

# Optimize
for step in tqdm(range(50)):

    # Compute the loss and step
    loss = optimizer.step(closure)
    print(f'Iter: {step}\t Loss = {loss.item()}')
    
    # Plot the resulting shape
    with torch.no_grad():
        plot_shape(
            model, 
            n_points=1000,
            output = os.path.join(OUTPUT_FOLDER, f'{step}.png')
        )
