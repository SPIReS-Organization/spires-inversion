"""I/O -> inversion contract boundary: speedy_invert_xarray must accept any
input the spires_contract spectra contract permits, regardless of dim order.

This exercises the conform_* wiring in speedy_invert_xarray: inputs handed over
in a non-canonical dimension order (and non-float64 dtype) must be conformed
internally and produce the canonical (y, x, 4) result.
"""
import numpy as np
import xarray as xr
import pytest

import spires_inversion
from spires_inversion.invert import speedy_invert_xarray
from spires_contract.spectra import (
    validate_target_spectra,
    validate_background_spectra,
    validate_solar_angles,
)

LUT_FILE = 'tests/data/lut_sentinel2b_b2to12_3um_dust.mat'


@pytest.fixture(scope='module')
def lut_dataarray():
    return spires_inversion.LutInterpolator(lut_file=LUT_FILE).to_xarray()


def _scene(n_bands):
    """A tiny 2x3 scene of target/background spectra and solar angles, built in
    CANONICAL (y, x, band) / (y, x) order with a band coordinate."""
    ny, nx = 2, 3
    rng = np.arange(ny * nx * n_bands, dtype=np.float64).reshape(ny, nx, n_bands)
    targets = xr.DataArray(
        0.3 + 0.01 * rng, dims=('y', 'x', 'band'),
        coords={'band': np.arange(n_bands)})
    backgrounds = xr.DataArray(
        0.1 + 0.01 * rng, dims=('y', 'x', 'band'),
        coords={'band': np.arange(n_bands)})
    angles = xr.DataArray(
        np.full((ny, nx), 50.0), dims=('y', 'x'))
    return targets, backgrounds, angles


def test_canonical_inputs_are_contract_valid(lut_dataarray):
    """Sanity: the fixtures we build are themselves contract-conforming."""
    n_bands = lut_dataarray.sizes['band']
    targets, backgrounds, angles = _scene(n_bands)
    # None of these should raise.
    validate_target_spectra(targets)
    validate_background_spectra(backgrounds)
    validate_solar_angles(angles)


def test_transposed_inputs_are_handled(lut_dataarray):
    """The contract permits any dim order; speedy_invert_xarray must accept a
    (band, y, x) target / float32 input and still return canonical (y, x, 4)."""
    n_bands = lut_dataarray.sizes['band']
    targets, backgrounds, angles = _scene(n_bands)

    # Hand over in a deliberately non-canonical order + float32 dtype.
    targets_t = targets.transpose('band', 'y', 'x').astype('float32')
    backgrounds_t = backgrounds.transpose('x', 'band', 'y').astype('float32')
    angles_t = angles.transpose('x', 'y').astype('float32')

    result = speedy_invert_xarray(targets_t, backgrounds_t, angles_t, lut_dataarray)

    # Canonical (y, x, 4) output regardless of input order.
    assert result.shape == (targets.sizes['y'], targets.sizes['x'], 4)
