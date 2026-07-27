# spires-inversion

[![PyPI version](https://badge.fury.io/py/spires.svg)](https://pypi.org/project/spires/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18747284.svg)](https://doi.org/10.5281/zenodo.18747284)
[![Documentation Status](https://readthedocs.org/projects/spires-inversion/badge/?version=latest)](https://spires-inversion.readthedocs.io/en/latest/?badge=latest)
[![Build and Test](https://github.com/NiklasPhabian/SpiPy/workflows/Build%20and%20Test/badge.svg)](https://github.com/NiklasPhabian/SpiPy/actions)
[![Python 3.9-3.14](https://img.shields.io/badge/python-3.9--3.14-blue.svg)](https://github.com/NiklasPhabian/SpiPy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[📦 View Source on GitHub](https://github.com/NiklasPhabian/SpiPy)** | **[📖 Documentation](https://spires-inversion.readthedocs.io)** | **[🐛 Report Issues](https://github.com/NiklasPhabian/SpiPy/issues)**

`spires-inversion` is a Python implementation of [SPIRES](https://ieeexplore.ieee.org/document/9290428) (Snow Property Inversion From Remote Sensing), originally implemented in MATLAB ([SPIRES GitHub repository](https://github.com/edwardbair/SPIRES)). It is the core inversion engine of the [SPIReS package family](https://github.com/SPIReS-Organization) and imports as `spires_inversion`.

## Overview

SPIRES retrieves snow properties (grain radius, light-absorbing-particle concentration, snow endmember fraction) from satellite multispectral imagery by inverting reflectance spectra using lookup tables generated from Mie-scattering theory.

**Key features:**
- Hybrid Python/C++ implementation for performance (3000x speedup over pure Python)
- Support for MODIS, Sentinel-2, and Landsat data
- SWIG bindings for optimized interpolation and optimization routines
- NLopt-based nonlinear optimization

## Installation

### Quick Install (PyPI)

```bash
pip install spires
```

**Note:** Pre-built binary wheels are available for Linux and macOS (Python 3.9-3.14). For other platforms or to build from source, see below.

### Install from Source
  
**Important:** Use conda-forge for all dependencies. The apt version of `nlopt` does not include required C++ headers.

```bash
# Install build tools and nlopt (required)
conda install -c conda-forge swig gxx gcc nlopt

# Install all dependencies (recommended)
conda install -c conda-forge numpy h5py scipy xarray netCDF4 gdal geopandas matplotlib tox sphinx dask jupyterlab pyproj
```

### Git LFS

This repository uses Git LFS for test data. Install Git LFS before cloning:

```bash
# macOS
brew install git-lfs

# Linux
sudo apt install git-lfs

# Initialize
git lfs install
```

### Build and Install

```bash
# Build SWIG extensions
python3 setup.py build_ext --inplace

# Install package
pip install .

# Or install with optional dependencies
pip install ".[dev,test,docs]"
```

## Usage

See the `examples/` folder for Jupyter notebooks with detailed use cases.

Basic usage:

```python
import spires_inversion
import numpy as np

# Load lookup table
interpolator = spires_inversion.LutInterpolator(
    lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat'
)

# Invert a single mixed spectrum
spectrum_target = np.array([0.3424, 0.366, 0.3624, 0.3893,
                            0.4162, 0.3957, 0.0704, 0.0627, 0.3792])
spectrum_background = np.array([0.0182, 0.0265, 0.0283, 0.0561,
                                0.0954, 0.1204, 0.1249, 0.0789, 0.1406])

fsnow, fshade, lap, grain_radius = spires_inversion.speedy_invert(
    spectrum_target=spectrum_target,
    spectrum_background=spectrum_background,
    solar_angle=55.73,
    interpolator=interpolator,
)
```

See [Getting Started](https://spires-inversion.readthedocs.io/en/latest/getting_started.html) for batch
processing, xarray, and Dask-parallel workflows.

## Development

### Building Wheels

Build a wheel for the active Python interpreter:

```bash
pip install build
python -m build --wheel
```

Build wheels for multiple Python versions using tox:

```bash
tox -e py39,py310,py311,py312
```

**Note:** When using pyenv, wheels for Python 3.9 may incorrectly build for x86 instead of arm64 on M1 Macs. Use a conda environment to build correctly.

### Building C++ Extensions Manually

The setuptools build process handles SWIG bindings automatically. To build manually:

```bash
cd spires
make
```

Or specify paths explicitly:

```bash
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
g++ -shared -o spires_module.so spires.cpp -I$NUMPY_INCLUDE
```

### Testing

Run doctests:

```bash
pytest --doctest-modules
```

### Documentation

Install documentation dependencies:

```bash
pip install ".[docs]"
```

Build documentation:

```bash
cd doc/
make html
```

## Lookup Tables and Test Data

### Lookup Tables

Simulated Mie-scattering snow reflectance lookup tables are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18701286.svg)](https://doi.org/10.5281/zenodo.18701286)

- **Sentinel-2**: `lut_sentinel2b_b2to12_3um_dust.mat` (70 MB)
- **HLS**: `lut_HLSS30_b1to13_3um_dust.mat` (101 MB)
- **MODIS**: `lut_modis_b1to7_3um_dust.mat` (537 MB)
- **Landsat OLI**: `lut_oli_b1to7_3um_dust.mat` (55 MB)

Download using the helper script:
```bash
python scripts/download_test_data.py --luts
```

Or download directly, e.g.:
```bash
curl -L -o lut_sentinel2b_b2to12_3um_dust.mat https://zenodo.org/records/18701286/files/lut_sentinel2b_b2to12_3um_dust.mat
```

**Note:** All LUTs above are also bundled in the repository via Git LFS — see
[tests/data/README.md](tests/data/README.md) for details.

### Test Data

Full-resolution test imagery for validation is available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18704072.svg)](https://doi.org/10.5281/zenodo.18704072)

- **Sentinel-2 reflectance**: `sentinel_r.nc` (1.4 GB, 921×1347 pixels)
- **Background reflectance**: `sentinel_r0.nc` (705 MB)

Small subsets suitable for CI/testing are included in the repository via Git LFS. See [tests/data/README.md](tests/data/README.md) for details.

## Performance

The C++ optimizations provide significant speedups over pure Python:

**Interpolation:** 3000x faster (1.07 ms → 309 ns)
- Pure Python RegularGridInterpolator: 1.07 ms
- Vectorized Python: 143 μs
- SWIG C++ (vectorized): 5.58 μs
- SWIG C++ (index lookup): 309 ns

**Spectrum Difference:** 1000x faster (1.1 ms → 1 μs)
- Pure Python: 1.1 ms
- With optimized interpolator: 3.8 μs
- C++ implementation: 1 μs

**Full Optimization:** 3000x faster (165 ms → 43 μs)
- Scipy optimization: 165 ms
- With optimized interpolator: 4.94 ms
- With C++ spectrum difference: 3.5 ms
- NLopt in C++: 43 μs

## Known Issues

- SLSQP solver doesn't work in the C++ implementation; using COBYLA instead
- SWIG interpolator and scipy's RegularGridInterpolator behave differently when coordinates aren't linspace
- COBYLA in scipy can't set `rhobeg` per dimension individually, requiring problem scaling

### Reparameterized algorithms (4, 5, 6) and grain-bound saturation

Algorithms 4-6 absorb the simplex constraint
(`f_sca + f_shade + f_bg = 1`, all ≥ 0) into a softmax reparameterization,
removing the need for inequality constraints. They differ in how they handle
the LUT box bounds on dust and grain:

- **Algorithms 4, 5 (full softmax):** sigmoid reparameterization on dust and
  grain, mapping the LUT range onto an unbounded variable in z-space.
- **Algorithm 6 (hybrid, recommended):** softmax for the fractions, but dust
  and grain stay in physical units and are *clipped* to the LUT range inside
  the objective — turning the bound into a true flat wall.

The hybrid is the recommended path. On a real 50×50 Sentinel-2 patch with exact
interpolation at the LUT upper bounds (algorithm benchmark, max_eval=100):

| Algorithm                     | Median residual | Grain ≥ 1199 µm | Time   | Speedup    |
|-------------------------------|-----------------|-----------------|--------|------------|
| 1: COBYLA                     | 0.1014          | 0 / 2500        | 171 ms | 1.0×       |
| 4: NELDERMEAD-softmax (full)  | 0.0951          | 4 / 2500        | 84 ms  | 2.0×       |
| **6: NELDERMEAD-hybrid**      | **0.0877**      | 376 / 2500      | **74 ms** | **2.3×** |

#### What grain-bound saturation actually means

The three algorithms produce different saturation counts for *different reasons*:

- **COBYLA (0/2500 at max_eval=100)** is implicitly regularized by the simplex inequality
  constraint; its search structure pulls toward the simplex interior, masking
  pixels whose true optimum lies at the LUT boundary.
- **Hybrid (376/2500 at max_eval=100)** reaches the boundary directly because
  the clip turns the bound into a flat wall. Its saturation set is still
  converging at 100 evaluations, but is effectively stable by 200 evaluations
  (517/2500 at max_eval=200 and 519/2500 at max_eval=500). Bounded reference
  searches found that most sampled saturated pixels had their best fit at the
  grain boundary, while a minority had near-equivalent interior optima.
- **Full softmax (4/2500 at max_eval=100, 446/2500 at max_eval=500)** has both
  a small set of genuine boundary cases *and* a drift mechanism: the sigmoid's
  derivative `d_grain/d_z_grain` vanishes as `z_grain → ∞`, so the optimizer
  keeps taking tiny improving steps that push `z_grain` upward without bound.
  Raising `max_eval` doesn't approach a fixed point — it accumulates more
  drift victims (442 additional pixels between 100 and 500 evaluations on this
  patch).

The diagnostic that separates "honest signal" from "drift artifact" is
**stability under max_eval**:

| Algorithm                  | @ max_eval=100   | @ max_eval=200     | @ max_eval=500     | Change 200 → 500 |
|----------------------------|------------------|--------------------|--------------------|------------------|
| 4: NELDERMEAD-softmax      | 4 / 2500 (0.16%) | 217 / 2500 (8.7%) | 446 / 2500 (17.8%) | **+229 (drift)** |
| 6: NELDERMEAD-hybrid       | 376 / 2500 (15%) | 517 / 2500 (20.7%) | 519 / 2500 (20.8%) | +2 (converged)   |

With valid endpoint interpolation, max_eval=100 is an early-stop
speed/accuracy choice for the hybrid rather than a converged saturation
diagnostic. Its set is essentially stable from 200 to 500; the full softmax's
set continues to grow. Continued growth after the initial search is the failure
mode — not the absolute saturation count.

The hybrid's clip-on-entry turns the LUT bound into a true flat plateau in the
objective: any value of dust or grain outside [min, max] maps to the same model
spectrum as the boundary itself, so further movement contributes nothing to the
residual and the simplex contracts and terminates. The clip introduces a
C^0-but-not-C^1 kink at the bound, which is benign for derivative-free solvers
(Nelder-Mead, COBYLA) but would need care for gradient-based methods.

**Recommendation:** use algorithm 6 (`NELDERMEAD-hybrid`) for new work. Treat
its saturated pixels as "the optimizer selected the LUT boundary" and flag
them downstream rather than assuming either exact physical truth or optimizer
noise. The default max_eval=100 favors speed; use at least 200 evaluations when
a stable boundary classification matters. If using algorithms 4 or 5, do not
raise `max_eval` above the default of 100 without a downstream filter, because
their saturation count grows with max_eval rather than converging.

### Cross-platform numerical reproducibility

Inversion results can differ by a few percent between Linux (x86_64, gcc) and macOS
(arm64, clang) for the same inputs. The cause is a combination of different ISAs
rounding transcendentals (`exp`, `pow`) at the last bit, different compilers and
libm implementations, and the fact that COBYLA is a derivative-free iterative
solver — tiny ULP-level differences in early evaluations can cascade and steer
the simplex toward a different local optimum on a flat region of the objective.
The recovered parameters still reproduce the observed spectrum within tolerance,
they just aren't bit-identical across platforms. Tests therefore assert residual
quality and physical plausibility rather than pinning optimizer coordinates.

## Roadmap

- [ ] Optimize inversion for single location over multiple timesteps (keep R_0 constant)
- [ ] Support xarray inputs for interpolator and spectra
- [ ] Add Landsat lookup tables
- [ ] Improve cloud masking workflows

## License

See LICENSE file for details.

## Citation

If you use this software, please cite the algorithm paper, software implementation, and any datasets you use:

**Algorithm:**
```bibtex
@article{bair2021spires,
  title={Snow Property Inversion From Remote Sensing (SPIReS): A Generalized Multispectral Unmixing Approach With Examples From MODIS and Landsat 8 OLI},
  author={Bair, E. H. and Stillinger, T. and Dozier, J.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={59},
  number={9},
  pages={7270--7284},
  year={2021},
  doi={10.1109/TGRS.2020.3040328}
}
```

**Software:**
```bibtex
@software{bair2026spipy,
  title={SpiPy: Python implementation of SPIRES snow property inversion},
  author={Bair, Edward H. and Griessbaum, Niklas},
  year={2026},
  url={https://github.com/NiklasPhabian/SpiPy},
  version={0.2.8},
  doi={10.5281/zenodo.18747284},
  note={See CITATION.cff for full metadata}
}
```

**Lookup Tables (if used):**
```bibtex
@dataset{bair2026spires_luts,
  author       = {Bair, Edward and Dozier, Jeff},
  title        = {{SPIRES} Snow Reflectance Lookup Tables},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18701286},
  url          = {https://doi.org/10.5281/zenodo.18701286}
}
```

**Test Data (if used):**
```bibtex
@dataset{griessbaum2026sentinel2_testdata,
  author       = {Griessbaum, Niklas},
  title        = {Sentinel-2 reflectance data for testing the {SpiPy} implementation of the {SPIRES} algorithm},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18704072},
  url          = {https://doi.org/10.5281/zenodo.18704072}
}
```

Alternatively, see [CITATION.cff](CITATION.cff) or use GitHub's "Cite this repository" feature.

## Funding

Development of this software was supported by:

**Contract:** W913E523C0002
**Program:** "Climate and natural hazards, snow-covered and mountain environment sensing research"
**Sponsor:** Broad Agency Announcement Program, Cold Regions Research and Engineering Laboratory
**Monitored by:** U.S. Army Engineer Research and Development Center, Hanover, NH 03755

**Distribution Statement:** Approved for public release; distribution is unlimited.
