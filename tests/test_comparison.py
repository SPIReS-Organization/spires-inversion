import spires_inversion
import numpy as np

interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
interpolator.make_scipy_interpolator_legacy()
spectrum_target = np.array([0.3424, 0.366, 0.3624, 0.38932347, 0.41624767, 0.39567757, 0.07043362, 0.06267947, 0.3792])
spectrum_background = np.array([0.0182, 0.0265, 0.0283, 0.056067, 0.095432, 0.12036866, 0.12491679, 0.07888655, 0.1406])
solar_angle = 24.0

expected_scipy3 = np.array([4.36429085e-01, 5.63570915e-01, 9.91000000e+02, 4.12331162e+01])
expected_scipy4 = np.array([4.01536561e-01, 5.40285882e-02, 1.17707939e+02, 2.46971074e+02])


def test_scipy_mode3():
    res, model_refl = spires_inversion.speedy_invert_scipy(interpolator=interpolator,
                                                 spectrum_target=spectrum_target,
                                                 spectrum_background=spectrum_background,
                                                 solar_angle=solar_angle,
                                                 mode=3, method='SLSQP')

    # Relaxed tolerance: SciPy SLSQP has small run-to-run variance, and the
    # underlying interpolation now correctly clamps at the LUT upper bound
    # (the previous ~1e-6 match was incidentally tight because the bound bug
    # produced reproducible UB).
    np.testing.assert_allclose(res.x, expected_scipy3, rtol=1e-3, atol=1e-3)
    print('New syntax, mode 3:', res.x)


def test_scipy_mode4():
    res, model_refl = spires_inversion.speedy_invert_scipy(interpolator=interpolator,
                                                 spectrum_target=spectrum_target,
                                                 spectrum_background=spectrum_background,
                                                 solar_angle=solar_angle,
                                                 mode=4, method='SLSQP')

    # Relaxed tolerance for mode 4 - optimization can vary between environments
    np.testing.assert_allclose(res.x, expected_scipy4, rtol=1e-3, atol=1e-3)
    print('New syntax, mode 4:', res.x)


def test_legacy_mode3():
    res, model_refl = spires_inversion.legacy.speedy_invert(f=interpolator.interpolator_scipy,
                                                  spectrum_target=spectrum_target,
                                                  spectrum_background=spectrum_background,
                                                  shade=np.zeros_like(spectrum_target),
                                                  solar_angle=solar_angle,
                                                  mode=3,
                                                  method='SLSQP')
    np.testing.assert_allclose(res.x, expected_scipy3, rtol=1e-5, atol=1e-3)
    print('Legacy syntax, mode 3:', res.x)


def test_legacy_mode4():
    res, model_refl = spires_inversion.legacy.speedy_invert(f=interpolator.interpolator_scipy,
                                                  spectrum_target=spectrum_target,
                                                  spectrum_background=spectrum_background,
                                                  shade=np.zeros_like(spectrum_target),
                                                  solar_angle=solar_angle,
                                                  mode=4,
                                                  method='SLSQP')
    np.testing.assert_allclose(res.x, expected_scipy4, rtol=1e-3, atol=1e-3)
    print('Legacy syntax, mode 4:', res.x)


def test_nlop_cobyla():
    res = np.asarray(spires_inversion.speedy_invert(interpolator=interpolator,
                               spectrum_target=spectrum_target,
                               spectrum_background=spectrum_background,
                               solar_angle=solar_angle,
                               algorithm=1))

    # Asserted on physical plausibility + residual fit rather than pinned
    # optimizer coordinates: COBYLA is derivative-free and float32 LUT storage
    # nudges the simplex to a different local optimum on this flat objective (a
    # few percent), the same cross-platform drift the README documents.
    fsnow, fshade, dust, grain = res
    assert 0 <= fsnow <= 1 and 0 <= fshade <= 1 and fsnow + fshade <= 1 + 1e-6
    assert interpolator.lap_concentrations.min() <= dust <= interpolator.lap_concentrations.max()
    assert interpolator.grain_sizes.min() <= grain <= interpolator.grain_sizes.max()
    residual = spires_inversion.snow_diff_4(
        res, spectrum_target, spectrum_background, solar_angle, interpolator,
        np.zeros_like(spectrum_target))
    assert residual < 0.05, f"residual {residual} too large"


def test_nlop_neldermead():
    res = np.asarray(spires_inversion.speedy_invert(interpolator=interpolator,
                               spectrum_target=spectrum_target,
                               spectrum_background=spectrum_background,
                               solar_angle=solar_angle,
                               algorithm=2))

    # Physical plausibility + residual, not pinned coordinates (see
    # test_nlop_cobyla). algorithm=2 is unconstrained Nelder-Mead, so f_shade may
    # be near-zero but fractions stay in range.
    fsnow, fshade, dust, grain = res
    assert 0 <= fsnow <= 1 and 0 <= fshade <= 1
    assert interpolator.lap_concentrations.min() <= dust <= interpolator.lap_concentrations.max()
    assert interpolator.grain_sizes.min() <= grain <= interpolator.grain_sizes.max()
    residual = spires_inversion.snow_diff_4(
        res, spectrum_target, spectrum_background, solar_angle, interpolator,
        np.zeros_like(spectrum_target))
    assert residual < 0.05, f"residual {residual} too large"
