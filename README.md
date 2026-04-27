# Sublinear Neural Networks for Convex Shape Optimization

Implementation of the paper:
**“Parametrizing Convex Sets Using Sublinear Neural Networks”**

This repository provides code to reproduce the numerical experiments from the paper, including:

* PDE-constrained shape optimization
* Mahler volume minimization
* Torsion-based optimization (MFS, Galerkin, PINNs)
* Minkowski problem

The method relies on sublinear neural networks to parameterize convex sets and uses automatic differentiation for computing geometric and PDE-related quantities .

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/EloiMartinet/SublinearNet.git
cd SublinearNet
pip install -e .
```

---

## 🚀 Running Experiments

All experiments are located in the `scripts/` folder.
Each experiment corresponds to a **single script**, except for the torsion benchmark.

---

## 1. Poisson Shape Optimization

📍 Paper: Section *4.1 – Optimization of a Poisson problem* 

```bash
python scripts/poisson_galerkin.py
```

This reproduces the optimization of shapes minimizing:
[
J(\Omega) = \int_\Omega u_\Omega
]
where (u_\Omega) solves a Poisson equation.

---

## 2. Mahler Volume Minimization

📍 Paper: Section *4.2 – Minimization of the Mahler volume* 

```bash
python scripts/mahler_volume.py
```

This experiment minimizes:

$$
\mathrm{Vol}(\Omega)\mathrm{Vol}(\Omega^\circ)
$$

among shapes satisfying certain symmetries.

---

## 3. Optimization of the gradient of the torsion function

📍 Paper: Section *4.3 – Maximization of the gradient of the torsion function* 

```bash
python scripts/max_torsion.py
```

This script explores torsion-based optimization objectives (related to Section 4.3).

---

## 4. Minkowski Problem

📍 Paper: Section *4.4 – Minkowski problem* 

```bash
python scripts/minkovski_problem.py
```

Solves a curvature prescription problem via neural parametrization.

---


## 5. Comparison with PINNs

📍 Maximizes the torsion function by three different methods (Appendix / Section C)

This benchmark compares:

* Method of Fundamental Solutions (MFS)
* Mesh-free Galerkin
* Physics-Informed Neural Networks (PINNs)

---

### ✅ Run all experiments (recommended)

```bash
python scripts/run_torsion_experiments.py
```

This will automatically:

1. Run `torsion_MFS.py`
2. Run `torsion_Galerkin.py`
3. Run `torsion_PINN.py` with:

   * `config_1`
   * `config_2`
   * `config_3`
4. Generate plots via `plot_torsion_experiments.py`

---

### ⚙️ Run individually (optional)

#### MFS

```bash
python scripts/torsion_MFS.py
```

#### Galerkin

```bash
python scripts/torsion_Galerkin.py
```

#### PINNs

```bash
python scripts/torsion_PINN.py --config config_1
python scripts/torsion_PINN.py --config config_2
python scripts/torsion_PINN.py --config config_3
```

👉 The three configs correspond to different hyperparameter settings used in the PINN comparison (see paper).

---

### 📊 Plotting results

```bash
python scripts/plot_torsion_experiments.py
```

⚠️ Requires running the experiments first.

---

### 🔍 Hyperparameter Search (optional)

```bash
python scripts/torsion_parameter_search.py
```

This reproduces the PINN hyperparameter search described in the paper.

---

## 📁 Output Files

Results are saved in:

```
res/
    csvs/                # numerical results
    torsion_Galerkin/    # Galerkin outputs
    torsion_MFS/         # MFS outputs
```

---

## 📖 Citation

If you use this code in you own research, please cite:

```bibtex
@article{sublinear_nn_convex_sets,
  title={Parametrizing Convex Sets Using Sublinear Neural Networks},
  author={Anonymous},
  journal={NeurIPS},
  year={2026}
}
```