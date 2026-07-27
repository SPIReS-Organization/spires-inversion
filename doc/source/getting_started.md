# Getting Started

## Installation

### Prerequisites

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
# Clone the repository
git clone https://github.com/NiklasPhabian/SpiPy.git
cd SpiPy

# Build SWIG extensions
python3 setup.py build_ext --inplace

# Install package
pip3 install .

# Or build a wheel
python -m build --wheel
pip3 install dist/*.whl
```

## Quick Start

### Basic Inversion Example

Here's a minimal example of inverting a single snow spectrum:

```python
import spires_inversion
import numpy as np

# Load the lookup table
interpolator = spires_inversion.interpolator.LutInterpolator(
    lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat'
)

# Define observation and background spectra
spectrum_target = np.array([0.3424, 0.366, 0.3624, 0.3893,
                           0.4162, 0.3957, 0.0704, 0.0627, 0.3792])
spectrum_background = np.array([0.0182, 0.0265, 0.0283, 0.0561,
                               0.0954, 0.1204, 0.1249, 0.0789, 0.1406])
solar_angle = 55.73  # degrees

# Run inversion. speedy_invert returns a positional tuple:
# (fsnow, fshade, lap, grain_radius)
fsnow, fshade, lap, grain_radius = spires_inversion.speedy_invert(
    spectrum_target=spectrum_target,
    spectrum_background=spectrum_background,
    solar_angle=solar_angle,
    interpolator=interpolator,
    algorithm=1  # LN_COBYLA
)

print(f"Snow endmember (fsnow): {fsnow:.3f}")
print(f"Grain radius: {grain_radius:.1f} μm")
print(f"LAP concentration: {lap:.1f} ppm")
```

### Canonical `SpiresData` workflow

For package-family workflows, prepare a `SpiresData` object with `spires-io`
and pass a canonical NetCDF reflectance LUT to the high-level entry point:

```python
import spires_inversion
import spires_io

data = spires_io.load("scene_config.json")

# Optional: clustering emits labels, counts, and inversion representatives.
data = spires_io.cluster(
    data,
    features=("reflectance",),
    reflectance_tol=0.02,
    apply_valid_inversion_mask=True,
)

data = spires_inversion.invert(
    data,
    lut="reflectance_lut.nc",
    algorithm=6,
    max_eval=100,
    apply_valid_inversion_mask=True,
    n_workers=1,
)
```

The function detects a complete canonical cluster schema automatically.
Partial cluster state is rejected, excluded labels (`-1`) remain NaN, and
clustered inputs must record the same effective valid-mask policy requested by
inversion. Direct and clustered paths emit `fsnow`, `fshade`,
`lap_concentration`, and `grain_radius` as canonical float32 `(y, x)` arrays.

The eager object path owns optional in-process threading through `n_workers`.
It defaults to one; use a larger value for a standalone single-scene job, but
keep it at one inside Dask or batch workers to avoid nested parallelism.

Clustering is an approximation rather than an accuracy-neutral optimization.
The Phase 3 VIIRS benchmark found a 1.53× end-to-end speedup at reflectance
tolerance `0.02` and 2.76× at `0.05`. The tighter tolerance was consistently
more accurate, but both had substantial upper-tail LAP and grain-radius
differences. Select a tolerance explicitly for the workflow rather than
treating either value as a universal default.

### Batch Processing

For processing multiple pixels or entire images, use the array-based functions:

```python
# Process a 2D array of spectra
results = spires_inversion.speedy_invert_array2d(
    spectra_targets=targets,      # shape: (ny, nx, n_bands)
    spectra_backgrounds=backgrounds,  # shape: (ny, nx, n_bands)
    obs_solar_angles=solar_angles,    # shape: (ny, nx)
    interpolator=interpolator,
    max_eval=100,
    algorithm=2  # LN_NELDERMEAD
)

# Extract results (columns: fsnow, fshade, lap, grain_radius)
fsnow = results[:, :, 0]
fshade = results[:, :, 1]
lap = results[:, :, 2]
grain_radius = results[:, :, 3]
```

### Using with xarray

For geospatial data with coordinates:

```python
import xarray as xr

# Load data as xarray DataArrays
targets = xr.open_dataarray('observations.nc')
backgrounds = xr.open_dataarray('background_r0.nc')
solar_angles = xr.open_dataarray('solar_angles.nc')

# Load LUT as xarray
lut = xr.open_dataarray('lut.nc')

# Run inversion
results = spires_inversion.speedy_invert_xarray(
    spectra_targets=targets,
    spectra_backgrounds=backgrounds,
    obs_solar_angles=solar_angles,
    lut_dataarray=lut
)
```

### Parallel Inversion with Dask

For datasets too large to fit in memory or that benefit from multi-core
processing (e.g. time series of full Sentinel-2 scenes), use the Dask-parallel
entry point. The C++ inversion releases the Python GIL, so a Dask client with
`threads_per_worker > 1` gives real parallel speedup while sharing one LUT
copy per worker process.

```python
import xarray as xr
from dask.distributed import Client
from spires_contract import validate_results

import spires_inversion

# Inputs as chunked DataArrays (time, y, x, band) etc.
targets = xr.open_zarr('sentinel2_data.zarr')['reflectance']
backgrounds = xr.open_dataarray('background_r0.nc')
solar_angles = xr.open_dataarray('solar_angles.nc')

interpolator = spires_inversion.LutInterpolator(
    lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat'
)

with Client(n_workers=4, threads_per_worker=4) as client:
    ds = spires_inversion.speedy_invert_dask(
        spectra_targets=targets,
        spectra_backgrounds=backgrounds,
        obs_solar_angles=solar_angles,
        interpolator=interpolator,
        client=client,
    )

    # The lazy result has already received non-computing structural validation.
    computed = ds.compute()
    for time_index in range(computed.sizes["time"]):
        validate_results(computed.isel(time=time_index, drop=True))

    # Explicit storage representation only; do not use it for postprocessing.
    encoded = spires_inversion.encode_results(computed)
    encoded.to_netcdf('inversion_results.nc')
```

Eager single-scene Dask results receive full numerical contract validation
before return. Lazy results and time stacks receive schema, dtype, coordinate,
and metadata validation without triggering computation; full numerical
validation is performed per eager `(y, x)` scene after materialization.

See `examples/05_sentinel_snow_inversion.ipynb` for a complete dask workflow.

## Understanding the Algorithm

SPIRES (SPectral Inversion of REflectance from Snow) retrieves snow properties by:

1. **Loading pre-computed lookup tables (LUTs)** - Generated from Mie scattering theory
2. **Defining a forward model** - Mixed pixel reflectance as a linear combination:
   ```
   R_mixed = fsnow * R_snow(lap, grain_radius, angle) +
             fshade * R_shade +
             (1 - fsnow - fshade) * R_background
   ```
3. **Optimizing parameters** - Minimizes difference between observed and modeled spectra
4. **Returning snow properties** - Snow endmember fraction, grain radius, LAP concentration

## Key Parameters

- **fsnow**: Fractional snow endmember (0-1) — the snow-covered fraction
- **grain_radius**: Effective snow grain radius (30-1200 μm)
- **lap**: Light-absorbing-particle concentration in snow (0-1000 ppm; the LUT
  is parameterized for dust)
- **R_0** (background): Snow-free reflectance spectrum

## Performance Notes

The C++ optimized version achieves dramatic speedups:
- Interpolation: **3000x faster** (1.07 ms → 309 ns)
- Full optimization: **3000x faster** (165 ms → 43 μs)

This enables processing entire satellite images (millions of pixels) in reasonable time.

## Next Steps

- See [Examples](examples.md) for complete workflow tutorials
- Check the [API Reference](reference.rst) for detailed function documentation
- Read the original paper: [Bair et al. (2021)](https://doi.org/10.1109/TGRS.2020.3040328)
