import numpy as np
import spires.core
import spires
import pytest

## Testing the .core functions

interpolator = spires.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat', )

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
    ret = spires.core.interpolate_all_array(lut=interpolator.reflectances,
                                            bands=interpolator.bands,
                                            solar_angles=interpolator.solar_angles,
                                            dust_concentrations=interpolator.dust_concentrations,
                                            grain_sizes=interpolator.grain_sizes,
                                            solar_angle=solar_angle,
                                            dust_concentration=dust_concentration,
                                            grain_size=grain_size)
    expected = np.array(
        [0.69418118, 0.72305336, 0.75899187, 0.76630307, 0.76921281, 0.75832135, 0.01766575, 0.02501143, 0.73101483])
    np.testing.assert_allclose(ret, expected, rtol=1e-5)


def test_spectrum_difference():
    x = [0.5, 0.01, dust_concentration, grain_size]
    ret = spires.core.spectrum_difference(x=x,
                                          spectrum_background=spectrum_background,
                                          spectrum_target=spectrum_target,
                                          spectrum_shade=spectrum_shade,
                                          solar_angle=solar_angle,
                                          bands=interpolator.bands,
                                          solar_angles=interpolator.solar_angles,
                                          dust_concentrations=interpolator.dust_concentrations,
                                          grain_sizes=interpolator.grain_sizes,
                                          lut=interpolator.reflectances)

    assert pytest.approx(ret, rel=1e-2) == 0.08295740267234748


def test_invert():
    x = spires.core.invert(spectrum_background=spectrum_background,
                           spectrum_target=spectrum_target,
                           spectrum_shade=spectrum_shade,
                           solar_angle=solar_angle,
                           bands=interpolator.bands,
                           solar_angles=interpolator.solar_angles,
                           dust_concentrations=interpolator.dust_concentrations,
                           grain_sizes=interpolator.grain_sizes,
                           lut=interpolator.reflectances,
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

    spires.core.invert_array1d(spectra_backgrounds=spectra_backgrounds,
                               spectra_targets=spectra_targets,
                               spectrum_shade=spectrum_shade,
                               obs_solar_angles=obs_solar_angles,
                               bands=interpolator.bands,
                               solar_angles=interpolator.solar_angles,
                               dust_concentrations=interpolator.dust_concentrations,
                               grain_sizes=interpolator.grain_sizes,
                               lut=interpolator.reflectances,
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

    results = spires.speedy_invert_array2d(spectra_targets=spectra_targets,
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
        residual = spires.core.spectrum_difference(
            x=results[row, 0],
            spectrum_background=background,
            spectrum_target=target,
            spectrum_shade=spectrum_shade,
            solar_angle=solar_angle,
            bands=interpolator.bands,
            solar_angles=interpolator.solar_angles,
            dust_concentrations=interpolator.dust_concentrations,
            grain_sizes=interpolator.grain_sizes,
            lut=interpolator.reflectances)
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
