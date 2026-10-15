import numpy as np
import pytest

import spires_inversion

xr = pytest.importorskip("xarray")
pytest.importorskip("dask")

interpolator = spires_inversion.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')

spectrum_target = np.array(
    [0.3424, 0.366, 0.3624, 0.38932347, 0.41624767, 0.39567757, 0.07043362, 0.06267947, 0.3792])
spectrum_background = np.array(
    [0.0182, 0.0265, 0.0283, 0.05606749, 0.09543234, 0.12036866, 0.12491679, 0.07888655, 0.1406])
solar_angle = 55.73733298

# Reference values come from test_swig.test_invert (same target/background/solar_angle, algorithm=1).
expected_per_pixel = np.array([4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02])


def _make_inputs(ny=2, nx=3, nt=None, chunks=None):
    # Imagery crosses the io->inversion boundary as float32 (contract dtype).
    n_bands = spectrum_target.size
    tgt32 = spectrum_target.astype(np.float32)
    bg32 = spectrum_background.astype(np.float32)
    # Geospatial y/x coords, as a real scene carries them (the results
    # contract requires y/x coords present on every output variable).
    ycoord = np.arange(ny, dtype=np.float64)
    xcoord = np.arange(nx, dtype=np.float64)
    if nt is None:
        targets = xr.DataArray(
            np.broadcast_to(tgt32, (ny, nx, n_bands)).copy(),
            dims=['y', 'x', 'band'],
            coords={'y': ycoord, 'x': xcoord})
        angles = xr.DataArray(
            np.full((ny, nx), solar_angle), dims=['y', 'x'],
            coords={'y': ycoord, 'x': xcoord})
    else:
        targets = xr.DataArray(
            np.broadcast_to(tgt32, (nt, ny, nx, n_bands)).copy(),
            dims=['time', 'y', 'x', 'band'],
            coords={'y': ycoord, 'x': xcoord})
        angles = xr.DataArray(
            np.full((nt, ny, nx), solar_angle), dims=['time', 'y', 'x'],
            coords={'y': ycoord, 'x': xcoord})

    backgrounds = xr.DataArray(
        np.broadcast_to(bg32, (ny, nx, n_bands)).copy(),
        dims=['y', 'x', 'band'],
        coords={'y': ycoord, 'x': xcoord})

    if chunks is not None:
        targets = targets.chunk(chunks)
        backgrounds = backgrounds.chunk({k: v for k, v in chunks.items() if k in backgrounds.dims})
        angle_chunks = {k: v for k, v in chunks.items() if k in angles.dims}
        if angle_chunks:
            angles = angles.chunk(angle_chunks)

    return targets, backgrounds, angles


def _assert_pixels_match(ds, ny, nx, nt=None):
    expected_vars = {'fsnow', 'fshade', 'lap_concentration', 'grain_radius'}
    assert set(ds.data_vars) == expected_vars

    if nt is None:
        expected_shape = (ny, nx)
        expected_dims = ('y', 'x')
    else:
        expected_shape = (nt, ny, nx)
        expected_dims = ('time', 'y', 'x')

    for i, name in enumerate(['fsnow', 'fshade', 'lap_concentration', 'grain_radius']):
        var = ds[name]
        assert var.shape == expected_shape
        assert tuple(var.dims) == expected_dims
        np.testing.assert_allclose(
            var.values,
            np.full(expected_shape, expected_per_pixel[i]),
            rtol=1e-4)

    for name in ['fsnow', 'fshade', 'lap_concentration', 'grain_radius']:
        assert ds[name].attrs.get('long_name')


def test_speedy_invert_dask_no_time():
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)
    if hasattr(res, 'compute'):
        res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3)


def test_speedy_invert_dask_with_time():
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, nt=2)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)
    spires_inversion.validate_result_structure(res)
    if hasattr(res, 'compute'):
        res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3, nt=2)
    from spires_contract import validate_results
    for time_index in range(res.sizes['time']):
        validate_results(res.isel(time=time_index, drop=True))


def test_speedy_invert_dask_chunked():
    chunks = {'y': 1, 'x': 2, 'band': -1}
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, chunks=chunks)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)
    # Lazy result: chunked input should yield a dask-backed dataset.
    assert any(res[v].chunks is not None for v in res.data_vars)
    spires_inversion.validate_result_structure(res)
    assert any(res[v].chunks is not None for v in res.data_vars)
    res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3)


def test_speedy_invert_dask_output_conforms_to_results_contract():
    from spires_contract import validate_results

    targets, backgrounds, angles = _make_inputs(ny=2, nx=3)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)
    if hasattr(res, 'compute'):
        res = res.compute()

    # Contract requires float32 result vars, y/x coords, and lap_type='dust'.
    for name in ['fsnow', 'fshade', 'lap_concentration', 'grain_radius']:
        assert res[name].dtype == np.float32
        assert 'y' in res[name].coords and 'x' in res[name].coords
    assert res['lap_concentration'].attrs.get('lap_type') == 'dust'

    # The canonical single-scene results validator must accept the output.
    validate_results(res)


def test_speedy_invert_dask_eager_output_is_validated():
    """The eager (non-dask) path validates its own output before returning."""
    from spires_contract import validate_results

    targets, backgrounds, angles = _make_inputs(ny=2, nx=3)
    # No .compute() call: an eager (numpy-backed) result must already conform.
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)
    validate_results(res)


def test_lazy_result_dtype_survives_compute():
    """The declared float32 dtype must still hold after .compute().

    apply_ufunc's ``output_dtypes`` is a declaration, not a cast: if the kernel
    returns float64 the dask array reports float32 while materializing float64,
    and ``.astype(np.float32)`` on such an array is a no-op (dask returns
    ``self`` when the requested dtype equals the declared one). See issue #15.
    """
    chunks = {'y': 1, 'x': 2, 'band': -1}
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, chunks=chunks)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)

    for name in ['fsnow', 'fshade', 'lap_concentration', 'grain_radius']:
        assert res[name].chunks is not None, f"{name} should be lazy"
        assert res[name].dtype == np.float32, f"{name} declared dtype"

    computed = res.compute()
    for name in ['fsnow', 'fshade', 'lap_concentration', 'grain_radius']:
        assert computed[name].dtype == np.float32, (
            f"{name} materialized as {computed[name].dtype}, but the lazy array "
            "declared float32")


def test_lazy_time_stacked_result_dtype_survives_compute():
    """Same declaration/kernel agreement for the 4D time-stacked branch."""
    chunks = {'y': 1, 'x': 2, 'band': -1}
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, nt=2, chunks=chunks)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)

    assert res['fsnow'].chunks is not None
    computed = res.compute()
    for name in ['fsnow', 'fshade', 'lap_concentration', 'grain_radius']:
        assert computed[name].dtype == np.float32, (
            f"{name} materialized as {computed[name].dtype}, but the lazy array "
            "declared float32")


@pytest.fixture(scope='module')
def local_client():
    """A small in-process distributed client for the scatter tests."""
    distributed = pytest.importorskip("dask.distributed")
    cluster = distributed.LocalCluster(
        n_workers=1, threads_per_worker=1, processes=False,
        dashboard_address=None)
    client = distributed.Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


def test_scatter_lut_keys_are_unique_across_calls(local_client):
    """Repeated scatters of the same LUT must not reuse dask keys.

    ``dask.array.from_array`` names are content-addressed, so scattering the
    same LUT twice produced identical keys. Identical keys across calls are the
    precondition for dask's release-vs-registration race: call N's futures
    release a key on the client's batched comm while call N+1 re-registers it
    over a ``scheduler.update_data`` RPC, with nothing ordering the two, giving
    ``FutureCancelledError: ... lost dependencies`` or a hang. See issue #16.
    """
    from spires_inversion.parallel import _import_dask, _scatter_lut

    dask, _ = _import_dask()

    first = _scatter_lut(local_client, interpolator.reflectances, dask)
    second = _scatter_lut(local_client, interpolator.reflectances, dask)

    assert first.name != second.name, (
        "two scatters of identical LUT content produced the same dask key "
        f"{first.name!r}; keys must be per-call unique")
    assert set(first.dask).isdisjoint(set(second.dask)), (
        "scattered graph keys overlap across calls")


def test_scatter_lut_round_trips_and_is_single_chunk(local_client):
    """The scattered LUT must equal its input and stay one chunk.

    The four LUT dims are *input core dims* of the ``apply_ufunc`` call, so a
    multi-chunk LUT raises ``Dimension 'dim1' with different chunksize
    present``. One chunk must be stated explicitly, not inherited from the
    global ``dask.config`` ``array.chunk-size`` default.
    """
    from spires_inversion.parallel import _import_dask, _scatter_lut

    dask, _ = _import_dask()

    reflectances = interpolator.reflectances
    scattered = _scatter_lut(local_client, reflectances, dask)

    assert scattered.npartitions == 1, (
        f"LUT scattered into {scattered.npartitions} chunks; the apply_ufunc "
        "core dims require exactly one")
    assert scattered.shape == reflectances.shape
    assert scattered.dtype == reflectances.dtype
    np.testing.assert_array_equal(scattered.compute(), reflectances)


def test_scatter_lut_single_chunk_under_small_chunk_size(local_client):
    """One chunk must hold even when array.chunk-size is tiny.

    ``from_array`` auto-chunks against ``dask.config`` ``array.chunk-size``
    (128 MiB default), so the single-chunk invariant held only by accident for
    LUTs smaller than that. A larger LUT, or a user who lowers the setting,
    broke a code path that never mentions chunking.
    """
    from spires_inversion.parallel import _import_dask, _scatter_lut

    dask, _ = _import_dask()

    with dask.config.set({'array.chunk-size': '1MiB'}):
        scattered = _scatter_lut(local_client, interpolator.reflectances, dask)

    assert scattered.npartitions == 1, (
        f"LUT scattered into {scattered.npartitions} chunks under a 1MiB "
        "array.chunk-size; the single-chunk invariant must be explicit")
    np.testing.assert_array_equal(
        scattered.compute(), interpolator.reflectances)


def test_speedy_invert_dask_with_scattered_lut(local_client):
    """End-to-end inversion with scatter_lut=True against a real client."""
    chunks = {'y': 1, 'x': 2, 'band': -1}
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, chunks=chunks)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=True, client=local_client, algorithm=1)
    res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3)
    for name in ['fsnow', 'fshade', 'lap_concentration', 'grain_radius']:
        assert res[name].dtype == np.float32


def test_encode_results_uses_canonical_fsnow_scale_name():
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3)
    results = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1)
    encoded = spires_inversion.encode_results(results, fsnow_scale=100)

    assert encoded['fsnow'].dtype == np.int8
    assert encoded['fsnow'].attrs['scale_factor'] == 0.01
    assert results['fsnow'].dtype == np.float32
