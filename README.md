# MDIF-toolkits
MDIF-toolkits is a analysis package for atomistic simulations with interfacial geometry.

# MDIF-toolkits

[![CI](https://github.com/cpark1602/MDIF-toolkits/actions/workflows/ci.yml/badge.svg)](https://github.com/cpark1602/MDIF-toolkits/actions)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A specialized scientific Python package designed to post-process, analyze, and extract structural and transport properties from Molecular Dynamics (MD) trajectories of complex electrified interfaces and slab geometries.

---

## Installation

### 1. Requirements
* **Python** $\ge$ 3.9
* Core dependencies (automatically resolved during installation): `numpy`, `scipy`, `pandas`

### 2. Local Installation (Development Mode)
To install the toolkit locally on your laptop or office machine so that changes you make to the source files are instantly active, clone the repository and run an editable installation:

```bash
git clone [https://github.com/cpark1602/MDIF-toolkits.git](https://github.com/cpark1602/MDIF-toolkits.git)
cd MDIF-toolkits
pip install -e .
```

---


## Available Analysis Modules

The package contains targeted workflows optimized for spatial sorting and time-correlation analysis of interfacial systems:

### Mass Density Profiling

Calculates the spatial mass distribution of different atomic species (e.g., water, ions) along a specific axis perpendicular to the slab geometry surface.


#### Usage
```bash
import MDAnalysis as mda
import scipy.constants

u_if = mda.Universe("run-pos.pdb", "run-pos.dcd")
boxX = 48.57
boxY = 15.667
boxZ = 15.076
box = [boxX, boxY, boxZ, 90, 90, 90]
u_if.dimensions = box
start_stop_step = [0, -1, 1]
print_results_path = "/results/"

import mass_density

dim = "x"
bin_size = 0.02
pbc = True
if_q0_nac = mass_density.Mass_density(
    u_if,
    box,
    print_results_path,
    pbc,
    bin_size,
    dim,
    start=start_stop_step[0],
    stop=start_stop_step[1],
    step=start_stop_step[2],
)

his_edges, number_density_O = if_q0_nac._get_densityProfile("O")
his_edges, number_density_H = if_q0_nac._get_densityProfile("H")
his_edges, number_density_Au = if_q0_nac._get_densityProfile("Au")
his_edges, number_density_Ne = if_q0_nac._get_densityProfile("Ne")
```


### Water Dipole Orientation Angles
Analyzes the structural ordering and polarization of water molecules in proximity to electrified interfaces by measuring the distribution of dipole vector angles relative to the surface normal.

### Hydrogen Bond Autocorrelation Function (ACF)
Tracks the dynamic lifetime, breaking, and forming kinetics of hydrogen bond networks near a surface using structural time-correlation frameworks.


### Ionic Conductivity
Evaluates charge transport dynamics across the electrolyte layer by parsing collective ionic current fluctuations under bias potentials.

### Mean Squared Displacement (MSD)

Calculates the translational diffusion coefficients of ionic species and solvent clusters in bulk or confined slab environments.

### Radial Distribution Function (RDF) in Slab Geometry
Computes a modified, spatially restricted g(r) function to evaluate pair correlation properties inside local thin-film slabs without bulk volume projection artifacts.


---

## Automated Code Quality & Testing

This project uses a modern CI/CD pipeline powered by GitHub Actions:

    Linting / Formatting: Regulated via ruff inside pre-commit hooks to guarantee clean code style.

    Unit Testing: Powered by pytest across Python 3.9, 3.10, and 3.11 matrices.

To run tests locally on your machine, simply execute:



---

## Contributors & Contact

    Developer: Chanbum Park (Sorbonne Université, CNRS)
               chanbum.park@sorbonne-universite.fr
               chanbum.park@ruhr-uni-bochum.de
               chanbum.park@theochem.ruhr-uni-bochum.de

    GitHub Repository: cpark1602/MDIF-toolkits


