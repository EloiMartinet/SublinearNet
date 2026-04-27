![Convex Shape Optimization](header.png)

# Convex Shape Optimization with Neural Networks

This repository implements a framework for **shape optimization over convex sets** using **neural networks**. The approach is then used to numerically explore **Blaschke–Santalò diagrams**, i.e. the image of shape functionals under convexity constraints.

The library is implemented almost entirely in **PyTorch**, with only a few lightweight dependencies.

> **Disclaimer**  
> This repository is linked to the paper **[Numerical exploration of the range of shape functionals using neural networks](https://www.arxiv.org/abs/2602.14881)**. It is strongly recommended to read the paper before running or modifying the code.

---

## :boom: Notebooks

In order to quickly see the capabilities of the library, you can directly copy and execute the following example notebooks (by increasing level of complexity):
- [Simulation of the isoperimetric problem](https://colab.research.google.com/drive/1IHtUUCk31hefELOGxjswUp08wRDh-bgy?usp=drive_link)
- [Minimization of Dirichlet eigenvalues](https://colab.research.google.com/drive/1zHei5Hz2LTWHSSRhA1oY19PLsYXCsvY3?usp=drive_link)
- [Simple Blaschke-Santalò diagram](https://colab.research.google.com/drive/1aokh44CqfrN53Rvl_zplbcMXPF9bvDF3?usp=drive_link)

Link to the folder: https://drive.google.com/drive/folders/12-tWL7n0nP48uCksgxboSXp1QX3xLcwN?usp=sharing

You can find the same scripts in the `examples/`folder.

You too can also use this library directly in Google Collab by executing
```bash
!pip install git+https://github.com/EloiMartinet/ConvexShapeOpt.git
```
in the first cell.


---

## :rocket: Installation

For local installation, please follow the following steps.

### 1. Clone the repository
```bash
git clone https://github.com/EloiMartinet/ConvexShapeOpt
```

### 2. Move to the project directory
```bash
cd ConvexShapeOpt
```

### 3. Install dependencies

For **PyTorch**, please consult the official [installation guide](https://pytorch.org/get-started/locally/).


> **Note**  
> CUDA is recommended for running the Blaschke–Santalò scripts.  
> The example scripts (simple shape optimization problems) run well on CPU.

Then, install the package in editable mode:
```bash
pip install -e .
```

---

## :computer_mouse: Usage

### Example: Isoperimetric Problem

From the root directory, run:
```bash
python3 examples/isoperimetric.py
```

This script simulates the minimization of the **perimeter** of a convex set with **unit measure**.

If the installation is successful, intermediate shapes are saved in:
```
res/isoperimetric/
```

The shapes should converge toward a **ball**.  
You are encouraged to open the script to see how simple it is to modify. For instance, changing
```python
DIM = 2
```
to
```python
DIM = 3
```
runs the same optimization in 3D.

---

### Blaschke–Santalò Diagrams

The scripts generating Blaschke–Santalò diagrams are located in the `scripts/` folder.

For example:
```bash
python3 scripts/diagram_APW_2d.py
```

This computes the diagram involving **area, perimeter, and moment of inertia** in 2D.

Results are organized as:
```
res/APW_2d/
├── shapes/     # Shape images and CSV of functional values
├── models/     # Saved neural network models
└── *.png       # Diagram snapshots
```

---

### Interactive Visualization

To visualize convex sets directly on the diagram, use the provided HTML tool.

From the root directory:
```bash
python3 -m http.server 8000
```

Then open:
```
http://localhost:8000/plot_diagram.html?folder=res/APW_2d/shapes
```

---

## :clipboard: Code Structure

The core of the library is located in:
```
src/shapes/invertible_nn.py
```

It implements the main model, **`ConvexDiffeo`**, whose `forward` method defines a **diffeomorphism from the unit ball to a convex set**.

This relies on the gauge function of a smoothed polygon, implemented in:
```
src/shapes/gauge_functions.py
```
via the `LSEGauge` class.

---

### Implemented Shape Quantities

The library includes differentiable implementations of the following quantities:

- Volume  
- Perimeter  
- Normal vector  
- Mean curvature  
- Willmore energy  
- Integral mean curvature  
- Torsional rigidity  
- Dirichlet and Neumann eigenvalues  

These computations heavily leverage **PyTorch automatic differentiation**.  
Please refer to the paper for further details.

> **Warning**  
> Use **double precision (`float64`)** for PDE-related quantities.

---

### Additional Modules

- `src/shapes/repulsion_energy.py`  
  Implements a simple **Riesz potential**, used throughout the scripts.

- `src/shapes/plot_utils.py`  
  Plotting utilities for **2D and 3D** visualization.

---

## :open_book: Documentation

Documentation is available here: https://convexshapeopt.readthedocs.io/en/latest/index.html

---

## :bookmark: Citation

If you use this code, please cite:
```
@misc{martinet2026numericalexplorationrangeshape,
      title={Numerical exploration of the range of shape functionals using neural networks}, 
      author={Eloi Martinet and Ilias Ftouhi},
      year={2026},
      eprint={2602.14881},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2602.14881}, 
}
```

---

## License

This project is released under the **MIT License**.