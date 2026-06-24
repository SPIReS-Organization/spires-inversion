"""Contract boundary behavior of speedy_invert_xarray.

The function validates its inputs against the spires_contract I/O->inversion
contract on entry (once per call), then passes them to the C++ kernel as-is. A
contract-conforming input is accepted; a misshaped one (wrong dim order, dtype,
missing coordinate) raises a clear ContractError here rather than a cryptic C++
failure.
"""
import numpy as np
import xarray as xr
import pytest

import spires_inversion
from spires_inversion.invert import speedy_invert_xarray
from spires_contract import ContractError
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
    CANONICAL (y, x, band) / (y, x) order, float64, with a band coordinate."""
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
    """The fixtures we build satisfy the contract validators (none raise)."""
    n_bands = lut_dataarray.sizes['band']
    targets, backgrounds, angles = _scene(n_bands)
    validate_target_spectra(targets)
    validate_background_spectra(backgrounds)
    validate_solar_angles(angles)


def test_consumer_accepts_contract_valid_inputs(lut_dataarray):
    """A canonical, contract-valid scene is processed to (y, x, 4)."""
    n_bands = lut_dataarray.sizes['band']
    targets, backgrounds, angles = _scene(n_bands)
    result = speedy_invert_xarray(targets, backgrounds, angles, lut_dataarray)
    assert result.shape == (targets.sizes['y'], targets.sizes['x'], 4)


def test_wrong_dim_order_raises_clear_error(lut_dataarray):
    """A transposed (band, y, x) target is rejected on entry with ContractError,
    not passed through to a cryptic C++ kernel failure."""
    n_bands = lut_dataarray.sizes['band']
    targets, backgrounds, angles = _scene(n_bands)
    targets_t = targets.transpose('band', 'y', 'x')  # legal data, wrong order
    with pytest.raises(ContractError):
        speedy_invert_xarray(targets_t, backgrounds, angles, lut_dataarray)
