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
NOISE_LEVEL = 0.01  # Fixed noise level
N_SAMPLES_LIST = np.logspace(1, 4, num=6, dtype=int)  # 100 → 10000
SHAPES = {'cube', 'ball', 'octahedron'}
N_RUNS = 100


# Set the output folder
OUTPUT_FOLDER = f'res/fit_fixed_noise_vary_samples'

# Create the output folder
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# shape -> n_samples -> list of runs
dict_histories = {
    shape: {n: [] for n in N_SAMPLES_LIST}
    for shape in SHAPES
}

for SHAPE in SHAPES:
    shape = ConvexDiffeo(input_size=DIM, gauge_function=SHAPE).to(device)

    for N_SAMPLES in N_SAMPLES_LIST:

        for run in range(N_RUNS):
            print(f'--- Shape={SHAPE}, N={N_SAMPLES}, Noise={NOISE_LEVEL:.2e}, Run={run} ---')

            model = ConvexDiffeo(
                input_size=DIM,
                n_unit=100,
                mode='gauge'
            ).to(device)

            # Sample data
            x = shape.sample_sphere(n_points=N_SAMPLES, random=True)
            noisy_samples = shape(x)
            noisy_samples += NOISE_LEVEL * torch.randn(noisy_samples.shape)

            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=0.1,
                max_iter=20,
                line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad()
                y_pred = model.sublinear_nn(noisy_samples)
                loss = torch.mean((y_pred - 1)**2)
                loss.backward()
                return loss

            # Optional: scale training steps with dataset size
            n_steps = int(50 * (N_SAMPLES / N_SAMPLES_LIST[0])**0.5)

            for step in range(n_steps):
                optimizer.step(closure)

            with torch.no_grad():
                # Evaluation on clean samples
                x = shape.sample_sphere(n_points=100_000, random=False)
                clean_samples = shape(x)
                y_pred = model.sublinear_nn(clean_samples)
                eval_loss = torch.sqrt(torch.mean((y_pred - 1)**2)).item()

                print(f'{eval_loss=}')
                dict_histories[SHAPE][N_SAMPLES].append(eval_loss)


def plot_results(dict_histories, sample_sizes, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": (6.5, 4.5),
        "lines.linewidth": 2,
    })

    fig, ax = plt.subplots()

    for shape, sample_dict in dict_histories.items():
        medians = []
        q25 = []
        q75 = []

        for n in sample_sizes:
            runs = np.array(sample_dict[n])

            medians.append(np.nanmedian(runs))
            q25.append(np.nanpercentile(runs, 25))
            q75.append(np.nanpercentile(runs, 75))

        medians = np.array(medians)
        q25 = np.array(q25)
        q75 = np.array(q75)

        # Median curve
        ax.plot(
            sample_sizes,
            medians,
            marker='o',
            label=shape.capitalize()
        )

        # Interquartile band
        ax.fill_between(
            sample_sizes,
            q25,
            q75,
            alpha=0.3
        )

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Shape Reconstruction vs Samples (Noise={NOISE_LEVEL:.1e})")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)

    plt.tight_layout()

    png_path = os.path.join(output_folder, "results.png")
    pdf_path = os.path.join(output_folder, "results.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plots to:\n- {png_path}\n- {pdf_path}")


# Call the function at the very end
plot_results(dict_histories, N_SAMPLES_LIST, OUTPUT_FOLDER)