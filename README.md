# MDIF-toolkits
MDIF-toolkits is a analysis package for atomistic simulations with interfacial geometry.

# MDIF-toolkits

[![CI](https://github.com/cpark1602/MDIF-toolkits/actions/workflows/ci.yml/badge.badge.svg)](https://github.com/cpark1602/MDIF-toolkits/actions)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A specialized scientific Python package designed to post-process, analyze, and extract structural and transport properties from Molecular Dynamics (MD) trajectories of complex electrified interfaces and slab geometries.

---

## 🚀 Installation

### 1. Requirements
* **Python** $\ge$ 3.9
* Core dependencies (automatically resolved during installation): `numpy`, `scipy`, `pandas`

### 2. Local Installation (Development Mode)
To install the toolkit locally on your laptop or office machine so that changes you make to the source files are instantly active, clone the repository and run an editable installation:

```bash
git clone [https://github.com/cpark1602/MDIF-toolkits.git](https://github.com/cpark1602/MDIF-toolkits.git)
cd MDIF-toolkits
pip install -e .
