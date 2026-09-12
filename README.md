<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 H. Peredo Fuentes & I. Martinez Villegas

Thermal Spray ROM Comparison — 1D Flattening Analysis (v1.0.0).
Zenodo DOI: 10.5281/zenodo.22729593
Original source: https://github.com/[your-username]/thermal_spray_rom

Documentation licensed under CC BY 4.0. Code licensed under MIT.
-->

# Thermal Spray ROM Comparison — 1D Flattening Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/1367580914.svg)](https://doi.org/10.5281/zenodo.22729593)

Reduced-order modeling (ROM) framework for particle flattening dynamics in
thermal spray, benchmarked against Nishioka and Fukumoto's (2000) experimental
Re–ξ data for Nickel particles (40, 80, and 120 µm).

---

## Overview

This repository contains the complete Python implementation used to generate
all results, figures, and tables in the manuscript:

> **"Reduced-Order Modeling of Particle Flattening Dynamics: A Benchmark
> Study of Krylov–Arnoldi Methods Using Nishioka's Experimental Data"**
> H. Peredo Fuentes, I. Martinez Villegas (2026)

The framework:

- Constructs a **full-order model** from Nishioka's discrete Re–ξ measurements
  using quadratic interpolation on a 200-point logarithmic grid with small
  Gaussian noise (σ = 0.002).
- Fits the interpolated signal with two continuous models:
  **Schiller–Naumann drag correlation** and a **3-parameter power law**.
- Applies **four ROM methods**:
  - **TD** — Time-Derivative Decomposition
  - **JK-P** — Jacobian-Projection
  - **JK-G_RK4** — Jacobian-Galerkin with RK4 integrator
  - **JK-G_Exp** — Jacobian-Galerkin with exponential integrator
- Evaluates **four validation metrics**: energy retention, peak error,
  phase error, and FRF H₂ error.
- Produces **convergence, FRF, and overlay plots** for all particle sizes.

---

## Method Summary

| Component | Implementation |
|-----------|----------------|
| Full-order model | Quadratic interpolation of Nishioka's 5 Re–ξ points onto N = 200 log-spaced Re values |
| Artificial Jacobian | Second-order diffusion operator (tridiagonal, negative diagonal) |
| Krylov basis | Arnoldi process with modified Gram–Schmidt |
| TD basis | Successive numerical derivatives of the signal, orthonormalised |
| ROM reduction dimension | q ∈ {2, 5, 10, 15, 20, 30, 40, 50}; final results at **q = 50** |
| Integrators | RK4 and matrix-exponential (Exp) |
| Theoretical models compared | Jones (1971), Madejski (1976), Mostaghimi (1996) |
| Mean particle velocity | v̄ = 105.0 m/s (from Nishioka's Fig. 8) |
| Physical frequency range | 420 Hz – 52.4 kHz |

---

## Requirements

- Python 3.12
- NumPy 1.26
- SciPy 1.11
- Matplotlib 3.8
- scikit-learn 1.3

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1 — Local execution (system Python)

```bash
python3 thermal_spray_rom_comparisonV951.py
```

### Option 2 — Google Colab (zero-install, cloud)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `thermal_spray_rom_comparisonV951.py`.
3. Run the notebook. All figures and tables will be generated in the
   `flattening_multi_size/` folder.

### Option 3 — Local execution (Python virtual environment, recommended)

```bash
git clone https://github.com/[your-username]/thermal_spray_rom.git
cd thermal_spray_rom
python3.12 -m venv venv
source venv/bin/activate           # Linux / macOS
# OR
venv\Scripts\activate              # Windows
pip install --upgrade pip
pip install -r requirements.txt
python3 thermal_spray_rom_comparisonV951.py
```

### Option 4 — Conda environment

```bash
conda env create -f environment.yml
conda activate thermal_spray_rom
python3 thermal_spray_rom_comparisonV951.py
```

### Option 5 — Docker (fully reproducible, OS-independent)

```bash
docker build -t thermal_spray_rom .
docker run --rm -v "$(pwd)/flattening_multi_size:/app/flattening_multi_size" thermal_spray_rom
```

### Option 6 — Binder (one-click cloud, no setup)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/[your-username]/thermal_spray_rom/main)

### Option 7 — GitHub Actions (automated reproducibility check)

Every push to the `main` branch triggers the workflow defined in
`.github/workflows/run_analysis.yml`.

---

## Output Files

After execution, the following files are generated in `flattening_multi_size/`:

| File | Description |
|------|-------------|
| `Overlay_Reconstructions.png` | Full signals vs ROM reconstructions for all particle sizes |
| `Overlay_Fits.png` | Schiller–Naumann and power-law fits against experimental data |
| `Metrics_Bar_Charts.png` | Energy, peak error, phase, and H₂ metrics per method |
| `Overlay_FRF.png` | Spatial-frequency FRF (cycles per Re) |
| `Overlay_FRF_vs_Hz_Physical.png` | Physical-frequency FRF (Hz) using Nishioka velocities |
| `Overlay_FRF_vs_Hz_Physical_SPL.png` | Same FRF converted to dB SPL |
| `Convergence_Energy.png` | Energy retention vs number of modes q |
| `Convergence_Errors.png` | Peak and H₂ error vs q |

Console output includes publication-ready LaTeX tables for:

- ROM performance metrics (all sizes and methods, q = 50)
- Fitting results (Schiller–Naumann vs power law, medium particles)
- Theoretical model errors (Jones, Madejski, Mostaghimi)
- Variation of ξ across the experimental Re range
- Convergence summary

---

## Reproducing Specific Figures and Tables

| **Figure / Table in Paper** | **File to Inspect** |
|-----------------------------|---------------------|
| Fig. 1 (ROM workflow) | TikZ diagram in the LaTeX source (no code) |
| Fig. 2 (Overlay reconstructions) | `Overlay_Reconstructions.png` |
| Fig. 3 (Fits overlay) | `Overlay_Fits.png` |
| Fig. 4 (Metrics bar charts) | `Metrics_Bar_Charts.png` |
| Fig. 5 (FRF vs Hz) | `Overlay_FRF_vs_Hz_Physical.png` |
| Fig. 6 (FRF vs Hz, SPL) | `Overlay_FRF_vs_Hz_Physical_SPL.png` |
| Fig. 7 (Convergence, energy) | `Convergence_Energy.png` |
| Fig. 8 (Convergence, errors) | `Convergence_Errors.png` |
| Table 1 (ROM performance) | Printed to console by `run_multi_size_analysis` |
| Table 2 (Fits) | Printed to console after the fitting block |
| Table 3 (Theoretical model errors) | Printed to console after the model comparison block |
| Table 4 (ξ variation) | Printed to console at the end of the script |
| Table 5 (Nishioka comparison) | Printed to console near the end |

---

## Key Results Reproduced

At q = 50, the code produces the following metrics (identical to the paper):

| Particle size | TD | JK-P | JK-G_RK4 | JK-G_Exp |
|---------------|-----|------|----------|----------|
| Small (40 µm) | 100 % energy, 0 % peak | 100 % energy, 0 % peak | 125.82 % energy, 22.61 % peak | 125.82 % energy, 22.60 % peak |
| Medium (80 µm) | 100 %, 0 % | 100 %, 0 % | 153.28 %, 48.40 % | 153.28 %, 48.39 % |
| Large (120 µm) | 100 %, 0 % | 100 %, 0 % | 215.09 %, 67.33 % | 215.09 %, 67.32 % |

Fitting metrics for the medium particles:

| Model | RMSE | R² |
|-------|------|----|
| Schiller–Naumann | 0.0356 | 0.9836 |
| Power law | 0.0298 | 0.9885 |

---

## Data Source

The experimental Re–ξ measurements used in this study are taken from:

> Nishioka, E., & Fukumoto, M. (2000). *Investigation of Flattening
> Behavior of Thermal Sprayed Particle Based on Measured Data*.
> Quarterly Journal of the Japan Welding Society, 18(2), 259–268.

The values are hard-coded in the script under the dictionary
`nishioka_exp_data`.

---

## Citation

If you use this code or the associated data in your own work, please cite
both the paper and the software archive:

### Paper

```bibtex
@article{PeredoFuentes2026,
  author  = {Peredo Fuentes, H. and Martinez Villegas, I.},
  title   = {Reduced-Order Modeling of Particle Flattening Dynamics:
             A Benchmark Study of Krylov-Arnoldi Methods Using
             Nishioka's Experimental Data},
  journal = {Journal of Thermal Spray Technology},
  year    = {2026}
}
```

### Software

```bibtex
@software{PeredoFuentes2026Code,
  author    = {Peredo Fuentes, H. and Martinez Villegas, I.},
  title     = {Thermal Spray ROM Comparison — 1D Flattening Analysis},
  version   = {v1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22729593},
  url       = {https://github.com/hperedo/thermal_spray_rom}
}
```

A `CITATION.cff` file is provided for automated citation managers.

---

## License

This project uses a **dual-license** structure:

- **Code** (all `.py` files): **MIT License** — see [`LICENSE`](LICENSE.txt) for the full text.
- **Documentation** (`README.md`, figures, and other non-code content):
  **Creative Commons Attribution 4.0 International (CC BY 4.0)** — see
  [`LICENSE-CC-BY-4.0.txt`](LICENSE-CC-BY-4.0.txt) for the full text.

### Attribution Notice

When reusing, modifying, or redistributing this work, you must include the
following attribution in a prominent location:

> **Peredo Fuentes, H. & Martinez Villegas, I. (2026).**
> *Thermal Spray ROM Comparison — 1D Flattening Analysis* (v1.0.0).
> Zenodo. DOI: [10.5281/zenodo.22729593](https://doi.org/10.5281/zenodo.22729593)
> Original source: <https://github.com/hperedo/thermal_spray_rom>
> Licensed under MIT (code) and CC BY 4.0 (documentation).

### Your Rights

You are free to:

- **Share** — copy and redistribute the material in any medium or format.
- **Adapt** — remix, transform, and build upon the material for any
  purpose, even commercially.

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to
  the license and the DOI, and indicate if changes were made. You may do so
  in any reasonable manner, but not in any way that suggests the licensor
  endorses you or your use.

---

## Contact

For questions, collaborations, or to report issues, please contact:

**Humberto Peredo Fuentes** — humberto.peredo@tec.mx
*Centro de Innovación en Manufactura Avanzada (CIMA),
Instituto Tecnológico de Estudios Superiores de Monterrey (ITESM),
Campus Querétaro, México*
