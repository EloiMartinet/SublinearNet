import subprocess
import sys

# Paths to the different scripts
SCRIPT_PINN = "scripts/torsion_PINN.py"  
SCRIPT_MFS = "scripts/torsion_MFS.py"
SCRIPT_GALERKIN = "scripts/torsion_Galerkin.py"
SCRIPT_PLOT = "scripts/plot_torsion_experiments.py"

# Configs for the PINN script
configs = ["config_1", "config_2", "config_3"]

# Run all
print(f"\n=== Running {SCRIPT_MFS} ===\n")

result = subprocess.run(
    [sys.executable, SCRIPT_MFS],
    check=True  # raises error if a run fails
)


print(f"\n=== Running {SCRIPT_GALERKIN} ===\n")

result = subprocess.run(
    [sys.executable, SCRIPT_GALERKIN],
    check=True  # raises error if a run fails
)


for cfg in configs:
    print(f"\n=== Running {SCRIPT_PINN} with {cfg} ===\n")

    result = subprocess.run(
        [sys.executable, SCRIPT_PINN, "--config", cfg],
        check=True  # raises error if a run fails
    )

print(f"\n=== Running {SCRIPT_PLOT} ===\n")

result = subprocess.run(
    [sys.executable, SCRIPT_PLOT],
    check=True  # raises error if a run fails
)

print("\nAll runs completed.")