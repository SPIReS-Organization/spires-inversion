"""Contract boundary behavior of speedy_invert_xarray.

The function validates its inputs against the spires_contract I/O->inversion
contract on entry (once per call), then passes them to the C++ kernel as-is. A
contract-conforming input is accepted; a misshaped one (wrong dim order, dtype,
missing coordinate) raises a clear ContractError here rather than a cryptic C++
failure.
"""
import importlib

import numpy as np
import xarray as xr
import pytest

import spires_inversion
from spires_inversion.invert import speedy_invert_xarray
from spires_contract import ContractError, SpiresData, validate_results
from spires_contract.spectra import (
    validate_target_spectra,
    validate_background_spectra,
    validate_solar_angles,
)

LUT_FILE = 'tests/data/lut_sentinel2b_b2to12_3um_dust.mat'
invert_module = importlib.import_module("spires_inversion.invert")


@pytest.fixture(scope='module')
def lut_dataarray():
    return spires_inversion.LutInterpolator(lut_file=LUT_FILE).to_xarray()


@pytest.fixture(scope='module')
def canonical_lut():
    return spires_inversion.load_matlab_reflectance_lut(
        LUT_FILE,
        lap_type='dust',
        lap_concentration_units='ppm',
        grain_radius_units='um',
        solar_angle_units='degrees',
    )


def _scene(n_bands):
    """A tiny 2x3 scene of target/background spectra and solar angles, built in
    CANONICAL (y, x, band) / (y, x) order, float32 (the contract dtype), with a
    band coordinate."""
    ny, nx = 2, 3
    rng = np.arange(ny * nx * n_bands, dtype=np.float32).reshape(ny, nx, n_bands)
    targets = xr.DataArray(
        (0.3 + 0.01 * rng).astype(np.float32), dims=('y', 'x', 'band'),
        coords={'band': np.arange(n_bands)})
    backgrounds = xr.DataArray(
        (0.1 + 0.01 * rng).astype(np.float32), dims=('y', 'x', 'band'),
        coords={'band': np.arange(n_bands)})
    angles = xr.DataArray(
        np.full((ny, nx), 50.0, dtype=np.float32), dims=('y', 'x'))
    return targets, backgrounds, angles


def _object_data(canonical_lut, *, clustered):
    bands = canonical_lut['reflectance']['band'].values
    y = np.array([100.0, 200.0])
    x = np.array([10.0, 20.0, 30.0])
    coords = {
        'y': y,
        'x': x,
        'band': bands,
        'spatial_ref': 0,
        'Projection': 0,
    }
    n_bands = bands.size

    first = np.full(n_bands, 0.2, dtype=np.float32)
    second = np.full(n_bands, 0.6, dtype=np.float32)
    target_values = np.stack(
        [first, first, np.full(n_bands, np.nan, dtype=np.float32),
         second, second, second]
    ).reshape(2, 3, n_bands)
    background_values = np.full((2, 3, n_bands), 0.1, dtype=np.float32)
    solar_values = np.array(
        [[40.0, 40.0, np.nan], [50.0, 50.0, 50.0]],
        dtype=np.float32,
    )
    scene = xr.Dataset(
        {
            'reflectance': xr.DataArray(
                target_values,
                dims=('y', 'x', 'band'),
                coords=coords,
            ),
            'solar_zenith': xr.DataArray(
                solar_values,
                dims=('y', 'x'),
                coords={'y': y, 'x': x},
            ),
        }
    )
    background = xr.DataArray(
        background_values,
        dims=('y', 'x', 'band'),
        coords=coords,
        name='background',
    )
    if not clustered:
        return SpiresData(scene=scene, background=background)

    cluster = np.array([0, 1], dtype=np.int64)
    labels = xr.DataArray(
        np.array([[0, 0, -1], [1, 1, 1]], dtype=np.int64),
        dims=('y', 'x'),
        coords={'y': y, 'x': x},
        attrs={
            'valid_inversion_mask_applied': False,
            'features': 'reflectance',
            'representative_method': 'cluster_mean',
            'reflectance_tol': '0.05',
        },
    )
    scene = scene.assign(
        cluster_label=labels,
        cluster_count=xr.DataArray(
            np.array([2, 3], dtype=np.int64),
            dims=('cluster',),
            coords={'cluster': cluster},
        ),
        cluster_representative_reflectance=xr.DataArray(
            np.stack([first, second]),
            dims=('cluster', 'band'),
            coords={'cluster': cluster, 'band': bands},
        ),
        cluster_representative_background=xr.DataArray(
            np.full((2, n_bands), 0.1, dtype=np.float32),
            dims=('cluster', 'band'),
            coords={'cluster': cluster, 'band': bands},
        ),
        cluster_representative_solar_zenith=xr.DataArray(
            np.array([40.0, 50.0], dtype=np.float32),
            dims=('cluster',),
            coords={'cluster': cluster},
        ),
    )
    return SpiresData(scene=scene, background=background)


def _fake_row_inversion(calls):
    def fake(*, spectra_targets, spectra_backgrounds, obs_solar_angles, **kwargs):
        calls.append(
            {
                'targets': np.array(spectra_targets, copy=True),
                'backgrounds': np.array(spectra_backgrounds, copy=True),
                'solar_angles': np.array(obs_solar_angles, copy=True),
                'max_eval': kwargs['max_eval'],
                'algorithm': kwargs['algorithm'],
            }
        )
        n_rows = spectra_targets.shape[0]
        return np.column_stack(
            [
                spectra_targets[:, 0],
                np.full(n_rows, 0.25),
                np.full(n_rows, 5.0),
                np.full(n_rows, 4.0),
            ]
        )

    return fake


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


def test_float64_lut_raises_clear_error(lut_dataarray):
    """A float64 LUT is rejected by the LUT contract (validate_lut) on entry,
    rather than reaching the float32-strict C++ typemap as a cryptic failure."""
    n_bands = lut_dataarray.sizes['band']
    targets, backgrounds, angles = _scene(n_bands)
    lut64 = lut_dataarray.astype(np.float64)
    with pytest.raises(ContractError):
        speedy_invert_xarray(targets, backgrounds, angles, lut64)


def test_clustered_object_inversion_dispatches_scatters_and_matches_direct(
    canonical_lut,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        invert_module,
        'speedy_invert_array1d',
        _fake_row_inversion(calls),
    )
    clustered = invert_module.invert(
        _object_data(canonical_lut, clustered=True),
        lut=canonical_lut,
        max_eval=37,
        algorithm=6,
    )
    direct = invert_module.invert(
        _object_data(canonical_lut, clustered=False),
        lut=canonical_lut,
        max_eval=37,
        algorithm=6,
    )

    assert calls[0]['targets'].shape == (2, canonical_lut.sizes['band'])
    assert calls[1]['targets'].shape == (5, canonical_lut.sizes['band'])
    assert calls[0]['max_eval'] == 37
    assert calls[0]['algorithm'] == 6
    xr.testing.assert_equal(clustered.results, direct.results)
    assert set(clustered.results.data_vars) == {
        'fsnow', 'fshade', 'lap_concentration', 'grain_radius'
    }
    for variable in clustered.results.data_vars:
        assert clustered.results[variable].dtype == np.float32
        assert np.isnan(clustered.results[variable].sel(y=100.0, x=30.0))
    assert clustered.results['grain_radius'].sel(y=200.0, x=10.0) == 16.0
    assert clustered.results.attrs['clustered_inversion'] == 1
    assert direct.results.attrs['clustered_inversion'] == 0
    assert clustered.results.attrs['inversion_algorithm'] == 6
    assert clustered.results.attrs['inversion_max_eval'] == 37
    assert clustered.results.attrs['valid_inversion_mask_available'] == 0
    assert clustered.results.attrs['valid_inversion_mask_applied'] == 0
    assert (
        clustered.results.attrs['reflectance_lut_identity']
        == 'lut_sentinel2b_b2to12_3um_dust.mat'
    )
    assert (
        clustered.results.attrs['reflectance_lut_normalization']
        == 'legacy_matlab_normalized'
    )
    assert clustered.results.attrs['clustering_features'] == 'reflectance'
    assert (
        clustered.results.attrs['clustering_representative_method']
        == 'cluster_mean'
    )
    assert clustered.results.attrs['clustering_reflectance_tol'] == '0.05'
    validate_results(clustered.results, scene=clustered.scene)


def test_clustered_object_inversion_rejects_mask_policy_mismatch(canonical_lut):
    data = _object_data(canonical_lut, clustered=True)
    scene = data.scene.copy(deep=False)
    scene['cluster_label'].attrs['valid_inversion_mask_applied'] = True

    with pytest.raises(ContractError, match='recluster'):
        invert_module.invert(
            data.assign_scene(scene),
            lut=canonical_lut,
        )


def test_clustered_object_inversion_supports_threaded_row_chunks(
    canonical_lut,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        invert_module,
        'speedy_invert_array1d',
        _fake_row_inversion(calls),
    )
    threaded = invert_module.invert(
        _object_data(canonical_lut, clustered=True),
        lut=canonical_lut,
        n_workers=2,
    )

    assert len(calls) == 2
    assert sorted(call['targets'].shape[0] for call in calls) == [1, 1]
    assert threaded.results['fsnow'].sel(y=100.0, x=10.0) == np.float32(0.2)
    assert threaded.results['fsnow'].sel(y=200.0, x=10.0) == np.float32(0.6)


@pytest.mark.parametrize('n_workers', [0, -1, 1.5, True])
def test_object_inversion_rejects_invalid_worker_count(canonical_lut, n_workers):
    with pytest.raises(ValueError, match='positive integer'):
        invert_module.invert(
            _object_data(canonical_lut, clustered=False),
            lut=canonical_lut,
            n_workers=n_workers,
        )


def test_clustered_object_inversion_rejects_partial_cluster_state(canonical_lut):
    data = _object_data(canonical_lut, clustered=True)
    scene = data.scene.drop_vars('cluster_count')

    with pytest.raises(ContractError, match='cluster_count'):
        invert_module.invert(
            data.assign_scene(scene),
            lut=canonical_lut,
        )


def test_clustered_object_inversion_handles_zero_clusters_without_kernel_call(
    canonical_lut,
    monkeypatch,
):
    data = _object_data(canonical_lut, clustered=True)
    scene = data.scene.drop_dims('cluster').copy(deep=False)
    cluster = np.array([], dtype=np.int64)
    bands = canonical_lut['reflectance']['band'].values
    scene['cluster_label'] = scene['cluster_label'].copy(
        data=np.full((2, 3), -1, dtype=np.int64)
    )
    scene['cluster_count'] = xr.DataArray(
        np.array([], dtype=np.int64),
        dims=('cluster',),
        coords={'cluster': cluster},
    )
    scene['cluster_representative_reflectance'] = xr.DataArray(
        np.empty((0, bands.size), dtype=np.float32),
        dims=('cluster', 'band'),
        coords={'cluster': cluster, 'band': bands},
    )
    scene['cluster_representative_background'] = xr.DataArray(
        np.empty((0, bands.size), dtype=np.float32),
        dims=('cluster', 'band'),
        coords={'cluster': cluster, 'band': bands},
    )
    scene['cluster_representative_solar_zenith'] = xr.DataArray(
        np.array([], dtype=np.float32),
        dims=('cluster',),
        coords={'cluster': cluster},
    )

    def fail_if_called(**kwargs):
        raise AssertionError('row inversion kernel must not run')

    monkeypatch.setattr(invert_module, 'speedy_invert_array1d', fail_if_called)
    inverted = invert_module.invert(data.assign_scene(scene), lut=canonical_lut)

    for variable in inverted.results.data_vars:
        assert np.isnan(inverted.results[variable].values).all()
