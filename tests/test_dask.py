import numpy as np
import pytest

import spires

xr = pytest.importorskip("xarray")
pytest.importorskip("dask")

interpolator = spires.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')

spectrum_target = np.array(
    [0.3424, 0.366, 0.3624, 0.38932347, 0.41624767, 0.39567757, 0.07043362, 0.06267947, 0.3792])
spectrum_background = np.array(
    [0.0182, 0.0265, 0.0283, 0.05606749, 0.09543234, 0.12036866, 0.12491679, 0.07888655, 0.1406])
solar_angle = 55.73733298

# Reference values come from test_swig.test_invert (same target/background/solar_angle, algorithm=1).
expected_per_pixel = np.array([4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02])


def _make_inputs(ny=2, nx=3, nt=None, chunks=None):
    n_bands = spectrum_target.size
    if nt is None:
        targets = xr.DataArray(
            np.broadcast_to(spectrum_target, (ny, nx, n_bands)).copy(),
            dims=['y', 'x', 'band'])
        angles = xr.DataArray(np.full((ny, nx), solar_angle), dims=['y', 'x'])
    else:
        targets = xr.DataArray(
            np.broadcast_to(spectrum_target, (nt, ny, nx, n_bands)).copy(),
            dims=['time', 'y', 'x', 'band'])
        angles = xr.DataArray(np.full((nt, ny, nx), solar_angle), dims=['time', 'y', 'x'])

    backgrounds = xr.DataArray(
        np.broadcast_to(spectrum_background, (ny, nx, n_bands)).copy(),
        dims=['y', 'x', 'band'])

    if chunks is not None:
        targets = targets.chunk(chunks)
        backgrounds = backgrounds.chunk({k: v for k, v in chunks.items() if k in backgrounds.dims})
        angle_chunks = {k: v for k, v in chunks.items() if k in angles.dims}
        if angle_chunks:
            angles = angles.chunk(angle_chunks)

    return targets, backgrounds, angles


def _assert_pixels_match(ds, ny, nx, nt=None):
    expected_vars = {'fsca', 'fshade', 'dust_concentration', 'grain_size'}
    assert set(ds.data_vars) == expected_vars

    if nt is None:
        expected_shape = (ny, nx)
        expected_dims = ('y', 'x')
    else:
        expected_shape = (nt, ny, nx)
        expected_dims = ('time', 'y', 'x')

    for i, name in enumerate(['fsca', 'fshade', 'dust_concentration', 'grain_size']):
        var = ds[name]
        assert var.shape == expected_shape
        assert tuple(var.dims) == expected_dims
        np.testing.assert_allclose(
            var.values,
            np.full(expected_shape, expected_per_pixel[i]),
            rtol=1e-4)

    for name in ['fsca', 'fshade', 'dust_concentration', 'grain_size']:
        assert ds[name].attrs.get('long_name')


def test_speedy_invert_dask_no_time():
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3)
    res = spires.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)
    if hasattr(res, 'compute'):
        res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3)


def test_speedy_invert_dask_with_time():
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, nt=2)
    res = spires.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)
    if hasattr(res, 'compute'):
        res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3, nt=2)


def test_speedy_invert_dask_chunked():
    chunks = {'y': 1, 'x': 2, 'band': -1}
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, chunks=chunks)
    res = spires.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)
    # Lazy result: chunked input should yield a dask-backed dataset.
    assert any(res[v].chunks is not None for v in res.data_vars)
    res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3)
