import numpy as np
import pytest

import spires_inversion
import spires_inversion.invert as invert
import spires_inversion.parallel as parallel

xr = pytest.importorskip("xarray")
pytest.importorskip("dask")


spectrum_target = np.array(
    [0.45, 0.46], dtype=np.float64)
spectrum_background = np.array(
    [0.10, 0.12], dtype=np.float64)
solar_angle = 30.0


class TinyInterpolator:
    def __init__(self):
        self.bands = np.array([1.0, 2.0], dtype=np.float64)
        self.solar_angles = np.array([30.0, 40.0], dtype=np.float64)
        self.dust_concentrations = np.array([0.0, 20.0], dtype=np.float64)
        self.grain_sizes = np.array([100.0, 500.0], dtype=np.float64)
        self.reflectances = np.ones((2, 2, 2, 2), dtype=np.float64) * 0.8


interpolator = TinyInterpolator()

expected_per_pixel = spires_inversion.speedy_invert_array2d(
    np.broadcast_to(spectrum_target, (1, 1, spectrum_target.size)).copy(),
    np.broadcast_to(spectrum_background, (1, 1, spectrum_background.size)).copy(),
    np.full((1, 1), solar_angle, dtype=np.float64),
    interpolator=interpolator,
    algorithm=1,
    max_eval=20,
)[0, 0]


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
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1, max_eval=20)
    if hasattr(res, 'compute'):
        res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3)


def test_speedy_invert_dask_with_time():
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, nt=2)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1, max_eval=20)
    if hasattr(res, 'compute'):
        res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3, nt=2)


def test_speedy_invert_dask_chunked():
    chunks = {'y': 1, 'x': 2, 'band': -1}
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, chunks=chunks)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1, max_eval=20)
    # Lazy result: chunked input should yield a dask-backed dataset.
    assert any(res[v].chunks is not None for v in res.data_vars)
    res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3)


def test_speedy_invert_dask_passes_spectrum_shade_and_valid_mask(monkeypatch):
    captured = {}

    def fake_speedy_invert_array2d(
        *,
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
        spectrum_shade,
        bands,
        solar_angles,
        dust_concentrations,
        grain_sizes,
        reflectances,
        max_eval,
        x0,
        algorithm,
        valid_mask,
        use_grouping,
        grouping_method,
        grouping_tolerance,
        grouping_reflectance_tol,
        grouping_background_tol,
        grouping_solar_zenith_tol,
    ):
        captured["spectrum_shade"] = spectrum_shade
        captured["valid_mask"] = valid_mask
        captured["use_grouping"] = use_grouping
        return np.zeros(spectra_targets.shape[:2] + (4,), dtype=np.float32)

    monkeypatch.setattr(parallel, "speedy_invert_array2d", fake_speedy_invert_array2d)

    spectra_targets = xr.DataArray(
        np.full((1, 1, 2), 0.2, dtype=np.float32),
        dims=("y", "x", "band"),
        coords={"y": [0], "x": [0], "band": ["b1", "b2"]},
    ).chunk({"y": 1, "x": 1, "band": -1})
    spectra_backgrounds = xr.DataArray(
        np.full((1, 1, 2), 0.1, dtype=np.float32),
        dims=("y", "x", "band"),
        coords=spectra_targets.coords,
    ).chunk({"y": 1, "x": 1, "band": -1})
    obs_solar_angles = xr.DataArray(
        np.full((1, 1), 30.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0], "x": [0]},
    ).chunk({"y": 1, "x": 1})
    valid_mask = xr.DataArray(np.array([[False]]), dims=("y", "x")).chunk({"y": 1, "x": 1})
    spectrum_shade = np.array([0.01, 0.02], dtype=np.float64)

    result = spires_inversion.speedy_invert_dask(
        spectra_targets=spectra_targets,
        spectra_backgrounds=spectra_backgrounds,
        obs_solar_angles=obs_solar_angles,
        interpolator=TinyInterpolator(),
        spectrum_shade=spectrum_shade,
        valid_mask=valid_mask,
        scatter_lut=False,
    ).compute()

    np.testing.assert_array_equal(captured["spectrum_shade"], spectrum_shade)
    np.testing.assert_array_equal(captured["valid_mask"], np.array([[False]]))
    assert captured["use_grouping"] is False
    assert result.attrs["grouping_enabled"] is False
    assert result.attrs["grouping_scope"] == "none"


def test_speedy_invert_array2d_grouping_broadcasts_results_and_skips_invalid(monkeypatch):
    captured = {}

    def fake_speedy_invert_array1d(
        *,
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
        spectrum_shade,
        bands,
        solar_angles,
        dust_concentrations,
        grain_sizes,
        reflectances,
        max_eval,
        x0,
        algorithm,
    ):
        captured["targets"] = spectra_targets.copy()
        captured["backgrounds"] = spectra_backgrounds.copy()
        captured["solar"] = obs_solar_angles.copy()
        captured["shade"] = spectrum_shade.copy()
        n = spectra_targets.shape[0]
        return np.column_stack(
            [
                np.arange(n, dtype=np.float64),
                np.arange(n, dtype=np.float64) + 10.0,
                np.arange(n, dtype=np.float64) + 20.0,
                np.arange(n, dtype=np.float64) + 30.0,
            ]
        )

    monkeypatch.setattr(invert, "speedy_invert_array1d", fake_speedy_invert_array1d)

    spectra_targets = np.array(
        [
            [[0.20, 0.30], [0.20, 0.30]],
            [[0.80, 0.90], [0.50, 0.60]],
        ],
        dtype=np.float64,
    )
    spectra_backgrounds = np.array(
        [
            [[0.10, 0.10], [0.10, 0.10]],
            [[0.40, 0.40], [0.20, 0.20]],
        ],
        dtype=np.float64,
    )
    obs_solar_angles = np.array([[30.0, 30.0], [40.0, 50.0]], dtype=np.float64)
    valid_mask = np.array([[True, True], [True, False]])
    spectrum_shade = np.array([0.01, 0.02])

    result = invert.speedy_invert_array2d(
        spectra_targets=spectra_targets,
        spectra_backgrounds=spectra_backgrounds,
        obs_solar_angles=obs_solar_angles,
        bands=np.array([1.0, 2.0]),
        solar_angles=np.array([30.0]),
        dust_concentrations=np.array([0.0]),
        grain_sizes=np.array([250.0]),
        reflectances=np.ones((2, 1, 1, 1), dtype=np.float64),
        spectrum_shade=spectrum_shade,
        valid_mask=valid_mask,
        use_grouping=True,
        grouping_method="first_pixel",
        grouping_tolerance=0.02,
    )

    assert captured["targets"].shape[0] == 2
    np.testing.assert_allclose(captured["targets"], np.array([[0.2, 0.3], [0.8, 0.9]]))
    np.testing.assert_allclose(captured["backgrounds"], np.array([[0.1, 0.1], [0.4, 0.4]]))
    np.testing.assert_allclose(captured["solar"], np.array([30.0, 40.0]))
    np.testing.assert_allclose(captured["shade"], spectrum_shade)
    np.testing.assert_allclose(result[0, 0], np.array([0.0, 10.0, 20.0, 30.0]))
    np.testing.assert_allclose(result[0, 1], np.array([0.0, 10.0, 20.0, 30.0]))
    np.testing.assert_allclose(result[1, 0], np.array([1.0, 11.0, 21.0, 31.0]))
    assert np.isnan(result[1, 1]).all()


def test_speedy_invert_dask_grouping_scene_scope_time_cube():
    chunks = {'time': 1, 'y': 2, 'x': 3, 'band': -1}
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, nt=2, chunks=chunks)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1, max_eval=20, use_grouping=True,
        grouping_scope="scene", grouping_method="first_pixel")
    res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3, nt=2)
    assert res.attrs["grouping_enabled"] is True
    assert res.attrs["grouping_scope"] == "scene"


def test_speedy_invert_dask_grouping_chunk_scope_static_r0():
    chunks = {'time': 2, 'y': 1, 'x': 2, 'band': -1}
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, nt=2, chunks=chunks)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1, max_eval=20, use_grouping=True,
        grouping_scope="chunk", grouping_method="first_pixel")
    res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3, nt=2)
    assert res.attrs["grouping_enabled"] is True
    assert res.attrs["grouping_scope"] == "chunk"


def test_speedy_invert_dask_grouping_chunk_scope_single_scene():
    chunks = {'y': 1, 'x': 2, 'band': -1}
    targets, backgrounds, angles = _make_inputs(ny=2, nx=3, chunks=chunks)
    res = spires_inversion.speedy_invert_dask(
        targets, backgrounds, angles, interpolator,
        scatter_lut=False, algorithm=1, max_eval=20, use_grouping=True,
        grouping_scope="chunk", grouping_method="first_pixel")
    res = res.compute()
    _assert_pixels_match(res, ny=2, nx=3)


def test_speedy_invert_dask_invalid_grouping_scope_raises():
    targets, backgrounds, angles = _make_inputs(ny=1, nx=1)
    with pytest.raises(ValueError, match="grouping_scope"):
        spires_inversion.speedy_invert_dask(
            targets,
            backgrounds,
            angles,
            interpolator,
            scatter_lut=False,
            use_grouping=True,
            grouping_scope="month",
        )
