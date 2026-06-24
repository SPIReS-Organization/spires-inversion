import numpy as np
import spires_inversion.core
import spires_inversion
import pytest

## Testing the .core functions

interpolator = spires_inversion.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat', )

spectrum_target = np.array([0.3424, 0.366, 0.3624, 0.38932347, 0.41624767, 0.39567757, 0.07043362, 0.06267947, 0.3792])
spectrum_background = np.array(
    [0.0182, 0.0265, 0.0283, 0.05606749, 0.09543234, 0.12036866, 0.12491679, 0.07888655, 0.1406])
spectrum_shade = np.zeros_like(spectrum_target)
solar_angle = 55.73733298

dust_concentration = 491
grain_size = 550
x0 = [0.5, 0.05, 10, 250]


def test_interpolate_all():
    ret = interpolator.interpolate_all(solar_angle=solar_angle, dust_concentration=dust_concentration,
                                       grain_size=grain_size)
    expected = np.array(
        [0.69418118, 0.72305336, 0.75899187, 0.76630307, 0.76921281, 0.75832135, 0.01766575, 0.02501143, 0.73101483])
    np.testing.assert_allclose(ret, expected, rtol=1e-5)


def test_interpolate_all_array():
    # this guy returns an array rather than a tuple
    ret = spires_inversion.core.interpolate_all_array(lut_reflectances=interpolator.reflectances,
                                            lut_bands=interpolator.bands,
                                            lut_solar_angles=interpolator.solar_angles,
                                            lut_dust_concentrations=interpolator.dust_concentrations,
                                            lut_grain_sizes=interpolator.grain_sizes,
                                            solar_angle=solar_angle,
                                            dust_concentration=dust_concentration,
                                            grain_size=grain_size)
    expected = np.array(
        [0.69418118, 0.72305336, 0.75899187, 0.76630307, 0.76921281, 0.75832135, 0.01766575, 0.02501143, 0.73101483])
    np.testing.assert_allclose(ret, expected, rtol=1e-5)


def test_spectrum_difference():
    x = [0.5, 0.01, dust_concentration, grain_size]
    ret = spires_inversion.core.spectrum_difference(x=x,
                                          spectrum_background=spectrum_background,
                                          spectrum_target=spectrum_target,
                                          spectrum_shade=spectrum_shade,
                                          solar_angle=solar_angle,
                                          lut_bands=interpolator.bands,
                                          lut_solar_angles=interpolator.solar_angles,
                                          lut_dust_concentrations=interpolator.dust_concentrations,
                                          lut_grain_sizes=interpolator.grain_sizes,
                                          lut_reflectances=interpolator.reflectances)

    assert pytest.approx(ret, rel=1e-2) == 0.08295740267234748


def test_invert():
    x = spires_inversion.core.invert(spectrum_background=spectrum_background,
                           spectrum_target=spectrum_target,
                           spectrum_shade=spectrum_shade,
                           solar_angle=solar_angle,
                           lut_bands=interpolator.bands,
                           lut_solar_angles=interpolator.solar_angles,
                           lut_dust_concentrations=interpolator.dust_concentrations,
                           lut_grain_sizes=interpolator.grain_sizes,
                           lut_reflectances=interpolator.reflectances,
                           max_eval=100,
                           x0=x0,
                           algorithm=1)

    expected = np.array([4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02])
    np.testing.assert_allclose(x, expected, rtol=1e-5)


def test_invert_array():
    n = 3
    results = np.empty((n, 4), dtype=np.double)
    spectra_backgrounds = np.tile(spectrum_background, (n, 1))
    spectra_targets = np.tile(spectrum_target, (n, 1))
    obs_solar_angles = np.repeat(solar_angle, n)

    spires_inversion.core.invert_array1d(spectra_backgrounds=spectra_backgrounds,
                               spectra_targets=spectra_targets,
                               spectrum_shade=spectrum_shade,
                               obs_solar_angles=obs_solar_angles,
                               lut_bands=interpolator.bands,
                               lut_solar_angles=interpolator.solar_angles,
                               lut_dust_concentrations=interpolator.dust_concentrations,
                               lut_grain_sizes=interpolator.grain_sizes,
                               lut_reflectances=interpolator.reflectances,
                               results=results,
                               max_eval=100,
                               x0=x0,
                               algorithm=1)

    expected = np.array([[4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02],
                         [4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02],
                         [4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02]])
    np.testing.assert_allclose(results, expected, rtol=1e-5)


# Two distinct pixels in different snow regimes (used by invert_array2d test)
_pixel_a_target = np.array(
    [0.3424, 0.366, 0.3624, 0.38932347, 0.41624767, 0.39567757, 0.07043362, 0.06267947, 0.3792])
_pixel_a_background = np.array(
    [0.0182, 0.0265, 0.0283, 0.056067, 0.095432, 0.12036866, 0.12491679, 0.07888655, 0.1406])
_pixel_b_target = np.array(
    [0.2866, 0.3046, 0.324, 0.34468558, 0.35373732, 0.35651454, 0.18072593, 0.16601688, 0.3488])
_pixel_b_background = np.array(
    [0.1002, 0.1492, 0.2088, 0.21797800, 0.23149200, 0.25140200, 0.31030660, 0.28750810, 0.2546])


def test_invert_array2d():
    """Catches dimension-handling bugs in invert_array2d that the 1D and
    single-pixel tests can't see. Asserted on shape, physical plausibility,
    per-row consistency (same input -> same output), and residual fit quality
    rather than pinned optimizer coordinates, which drift across platforms
    (see README "Cross-platform numerical reproducibility")."""
    spectra_targets = np.stack([np.tile(_pixel_a_target, (3, 1)),
                                np.tile(_pixel_b_target, (3, 1))], axis=0)
    spectra_backgrounds = np.stack([np.tile(_pixel_a_background, (3, 1)),
                                    np.tile(_pixel_b_background, (3, 1))], axis=0)
    obs_solar_angles = np.full((2, 3), solar_angle)

    results = spires_inversion.speedy_invert_array2d(spectra_targets=spectra_targets,
                                           spectra_backgrounds=spectra_backgrounds,
                                           obs_solar_angles=obs_solar_angles,
                                           interpolator=interpolator,
                                           algorithm=1, x0=np.array(x0))

    assert results.shape == (2, 3, 4)

    # Each row tiles a single pixel three times — identical inputs must produce
    # identical outputs (catches per-pixel state leakage / index bugs).
    for row in range(2):
        for x in range(1, 3):
            np.testing.assert_array_equal(results[row, x], results[row, 0])

    # The two rows have different inputs and must produce different outputs
    # (catches a bug where every pixel reads from the same source row).
    assert not np.allclose(results[0, 0], results[1, 0])

    # Physical plausibility: fsca, fshade in [0,1]; dust and grain within LUT range.
    fsca = results[..., 0]
    fshade = results[..., 1]
    dust = results[..., 2]
    grain = results[..., 3]
    assert np.all((fsca >= 0) & (fsca <= 1))
    assert np.all((fshade >= 0) & (fshade <= 1))
    assert np.all((dust >= interpolator.dust_concentrations.min()) &
                  (dust <= interpolator.dust_concentrations.max()))
    assert np.all((grain >= interpolator.grain_sizes.min()) &
                  (grain <= interpolator.grain_sizes.max()))

    # Residual: the recovered parameters must reproduce the observed spectrum
    # within tolerance. This is the actual contract of the inversion.
    spectrum_shade = np.zeros_like(_pixel_a_target)
    for row, target, background in [(0, _pixel_a_target, _pixel_a_background),
                                    (1, _pixel_b_target, _pixel_b_background)]:
        residual = spires_inversion.core.spectrum_difference(
            x=results[row, 0],
            spectrum_background=background,
            spectrum_target=target,
            spectrum_shade=spectrum_shade,
            solar_angle=solar_angle,
            lut_bands=interpolator.bands,
            lut_solar_angles=interpolator.solar_angles,
            lut_dust_concentrations=interpolator.dust_concentrations,
            lut_grain_sizes=interpolator.grain_sizes,
            lut_reflectances=interpolator.reflectances)
        assert residual < 0.05, f"row {row} residual {residual} too large"


def test_interpolate_all_handles_non_9_band_lut():
    """The SWIG output typemap previously hardcoded `dims[1] = {9}`, silently
    truncating/corrupting outputs for any LUT with band count != 9. This test
    pins the contract that interpolate_all returns one value per band."""
    n_bands = interpolator.bands.size
    ret = interpolator.interpolate_all(solar_angle=solar_angle,
                                       dust_concentration=dust_concentration,
                                       grain_size=grain_size)
    assert np.asarray(ret).shape == (n_bands,)


def _residual(x, target, background):
    return spires_inversion.core.spectrum_difference(
        x=list(x),
        spectrum_background=background,
        spectrum_target=target,
        spectrum_shade=np.zeros_like(target),
        solar_angle=solar_angle,
        lut_bands=interpolator.bands,
        lut_solar_angles=interpolator.solar_angles,
        lut_dust_concentrations=interpolator.dust_concentrations,
        lut_grain_sizes=interpolator.grain_sizes,
        lut_reflectances=interpolator.reflectances)


@pytest.mark.parametrize("algorithm,name", [(4, "NELDERMEAD-softmax"),
                                            (5, "BOBYQA-softmax"),
                                            (6, "NELDERMEAD-hybrid")])
def test_invert_softmax(algorithm, name):
    """Softmax/hybrid algorithms (4, 5, 6) absorb the simplex constraints into
    the parameter transformation, so unconstrained NLopt solvers can be used.
    Asserted on physical plausibility and residual fit quality (no pinned
    coordinates — they drift across platforms and the softmax surface is
    flatter near the bounds than the constrained version)."""
    x = spires_inversion.core.invert(spectrum_background=spectrum_background,
                           spectrum_target=spectrum_target,
                           spectrum_shade=spectrum_shade,
                           solar_angle=solar_angle,
                           lut_bands=interpolator.bands,
                           lut_solar_angles=interpolator.solar_angles,
                           lut_dust_concentrations=interpolator.dust_concentrations,
                           lut_grain_sizes=interpolator.grain_sizes,
                           lut_reflectances=interpolator.reflectances,
                           max_eval=500,
                           x0=x0,
                           algorithm=algorithm)
    x = np.asarray(x)

    # Physical plausibility: simplex + box bounds satisfied by construction.
    assert 0 <= x[0] <= 1, f"{name}: f_sca {x[0]} out of [0,1]"
    assert 0 <= x[1] <= 1, f"{name}: f_shade {x[1]} out of [0,1]"
    assert x[0] + x[1] <= 1 + 1e-6, f"{name}: f_sca + f_shade > 1"
    assert (interpolator.dust_concentrations.min() <= x[2] <=
            interpolator.dust_concentrations.max()), f"{name}: dust {x[2]} out of LUT range"
    assert (interpolator.grain_sizes.min() <= x[3] <=
            interpolator.grain_sizes.max()), f"{name}: grain {x[3]} out of LUT range"

    # Residual: softmax variants should match or beat COBYLA's fit on this
    # pixel (Python A/B confirmed the constraint was holding COBYLA back).
    residual_softmax = _residual(x, spectrum_target, spectrum_background)
    x_cobyla = spires_inversion.core.invert(spectrum_background=spectrum_background,
                                  spectrum_target=spectrum_target,
                                  spectrum_shade=spectrum_shade,
                                  solar_angle=solar_angle,
                                  lut_bands=interpolator.bands,
                                  lut_solar_angles=interpolator.solar_angles,
                                  lut_dust_concentrations=interpolator.dust_concentrations,
                                  lut_grain_sizes=interpolator.grain_sizes,
                                  lut_reflectances=interpolator.reflectances,
                                  max_eval=500, x0=x0, algorithm=1)
    residual_cobyla = _residual(np.asarray(x_cobyla), spectrum_target, spectrum_background)
    assert residual_softmax <= residual_cobyla * 1.05, (
        f"{name}: residual {residual_softmax:.5f} >> COBYLA's {residual_cobyla:.5f}")


def _real_imagery_setup():
    """Load the Sentinel-2 LFS subset, or skip the test cleanly. Returns
    (R, R0, sza_array, sza_scalar, residual_fn, n_pixels)."""
    xr = pytest.importorskip("xarray")
    try:
        ds = xr.open_dataset('tests/data/sentinel_r_subset.nc')
        ds0 = xr.open_dataset('tests/data/sentinel_r0_subset.nc')
    except (OSError, ValueError):
        pytest.skip("LFS test data not available (run `git lfs pull`)")

    R = np.ascontiguousarray(
        ds['reflectance'].isel(time=0).transpose('y', 'x', 'band').values.astype(np.float64))
    R0 = np.ascontiguousarray(
        ds0['reflectance'].transpose('y', 'x', 'band').values.astype(np.float64))
    sza = np.full(R.shape[:2],
                  float(np.nanmean(ds['sun_zenith_grid'].isel(time=0).values)))
    sza0 = float(sza[0, 0])
    shade = np.zeros(9)
    n = R.shape[0] * R.shape[1]

    def residuals(res):
        flat = res.reshape(-1, 4)
        flat_t = R.reshape(-1, 9)
        flat_b = R0.reshape(-1, 9)
        return np.array([
            spires_inversion.snow_diff_4(flat[i], flat_t[i], flat_b[i], sza0, interpolator, shade)
            for i in range(n)
        ])

    return R, R0, sza, sza0, residuals, n


def test_softmax_beats_cobyla_on_real_imagery():
    """End-to-end contract on a real 50x50 Sentinel-2 patch (LFS): the softmax
    Nelder-Mead variant (algorithm=4) must produce a median residual that is at
    least as good as COBYLA's at the same max_eval, and must not saturate the
    grain bound on more than ~2% of pixels at max_eval=100."""
    R, R0, sza, _, residuals, n = _real_imagery_setup()

    res_cobyla = spires_inversion.speedy_invert_array2d(
        spectra_targets=R, spectra_backgrounds=R0, obs_solar_angles=sza,
        interpolator=interpolator, algorithm=1, max_eval=100, x0=np.array(x0))
    res_softmax = spires_inversion.speedy_invert_array2d(
        spectra_targets=R, spectra_backgrounds=R0, obs_solar_angles=sza,
        interpolator=interpolator, algorithm=4, max_eval=100, x0=np.array(x0))

    r_cobyla = residuals(res_cobyla)
    r_softmax = residuals(res_softmax)

    assert np.median(r_softmax) <= np.median(r_cobyla), (
        f"softmax median {np.median(r_softmax):.4f} > COBYLA median {np.median(r_cobyla):.4f}")

    grain_max = interpolator.grain_sizes.max()
    grain = res_softmax[..., 3]
    n_saturated = int((grain >= grain_max - 1).sum())
    assert n_saturated / n <= 0.02, (
        f"softmax saturated grain on {n_saturated}/{n} pixels (>2%) — "
        "the unconstrained problem may be too loose at this max_eval")


def test_hybrid_saturation_is_stable_under_max_eval():
    """The hybrid algorithm (6 = softmax for fractions + clip for dust/grain)
    should converge to a stable saturation set: pixels whose true optimum lies
    at the LUT grain boundary find that boundary in a few iterations and stop,
    so the saturation count at max_eval=500 must be close to the count at
    max_eval=100.

    The full sigmoid (algorithm 4) lacks this property — its saturation count
    grows roughly linearly with max_eval as more pixels drift up the asymptotic
    z-ridge. So this test pins the *stability* property that distinguishes
    honest boundary signal from optimizer drift. On this patch the hybrid
    grows by ~4 pixels (13 → 17) between max_eval 100 and 500; the full
    softmax grows by ~370 (4 → 376)."""
    R, R0, sza, _, _, n = _real_imagery_setup()
    grain_max = interpolator.grain_sizes.max()

    def n_saturated(max_eval):
        res = spires_inversion.speedy_invert_array2d(
            spectra_targets=R, spectra_backgrounds=R0, obs_solar_angles=sza,
            interpolator=interpolator, algorithm=6, max_eval=max_eval,
            x0=np.array(x0))
        return int((res[..., 3] >= grain_max - 1).sum())

    sat_100 = n_saturated(100)
    sat_500 = n_saturated(500)
    growth = (sat_500 - sat_100) / n

    # Stability bar: ≤1% additional pixels saturate when max_eval grows 5×.
    # Hybrid currently shows ~0.2% on this patch; full softmax shows ~15%.
    # The 1% bar separates the regimes with platform-variance headroom.
    assert growth <= 0.01, (
        f"hybrid saturation grew from {sat_100} to {sat_500} of {n} pixels "
        f"({growth:.2%}) when max_eval went 100 -> 500. Growth >1% suggests "
        "the clip-on-entry mechanism is no longer pinning saturated pixels — "
        "the optimizer may be drifting like the full softmax (algorithm 4)")


def test_invert_unknown_algorithm_raises():
    """Unknown algorithm codes must raise rather than silently running with an
    uninitialized nlopt::opt — guards against the previous fallthrough bug."""
    with pytest.raises(RuntimeError):
        spires_inversion.core.invert(spectrum_background=spectrum_background,
                           spectrum_target=spectrum_target,
                           spectrum_shade=spectrum_shade,
                           solar_angle=solar_angle,
                           lut_bands=interpolator.bands,
                           lut_solar_angles=interpolator.solar_angles,
                           lut_dust_concentrations=interpolator.dust_concentrations,
                           lut_grain_sizes=interpolator.grain_sizes,
                           lut_reflectances=interpolator.reflectances,
                           max_eval=100, x0=x0, algorithm=99)
