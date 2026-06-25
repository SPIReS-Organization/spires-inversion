import spires_inversion.interpolator
import spires_inversion.core
from spires_inversion.grouping import group_spectra_block, scatter_group_results_block
from spires_contract.spectra import (
    validate_target_spectra,
    validate_background_spectra,
    validate_solar_angles,
)
import numpy as np
import scipy


def speedy_invert(spectrum_target, spectrum_background, solar_angle, spectrum_shade=None,
                  bands=None, solar_angles=None, dust_concentrations=None, grain_sizes=None, reflectances=None,
                  interpolator=None, lut_dataarray=None, max_eval=100, x0=np.array([0.5, 0.05, 10, 250]), algorithm=2):
    """
    Inverts the snow reflectance spectrum using nonlinear optimization.

    Parameters
    ----------
    spectrum_target : numpy.ndarray
        The mixed spectrum to invert. Must be same length as `spectrum_background`.
        Must have same band order as `spectrum_background` and `bands`.
    spectrum_background : numpy.ndarray
        The background (snow-free, R_0) spectrum.
    solar_angle : float
        The solar zenith angle of the spectrum target (degrees).
    spectrum_shade : numpy.ndarray, optional
        The ideal shaded spectrum. Must be same length as `spectrum_target`.
        If None, uses zeros (default: None).
    bands : numpy.ndarray, optional
        Band wavelength coordinates of reflectances. Required if interpolator not provided.
    solar_angles : numpy.ndarray, optional
        Solar angle coordinates of reflectances. Required if interpolator not provided.
    dust_concentrations : numpy.ndarray, optional
        Dust concentration coordinates of reflectances (ppm). Required if interpolator not provided.
    grain_sizes : numpy.ndarray, optional
        Grain size coordinates of reflectances (μm). Required if interpolator not provided.
    reflectances : numpy.ndarray, optional
        4D snow reflectance lookup table with dimensions (bands, solar_angles,
        dust_concentrations, grain_sizes). Required if interpolator not provided.
    interpolator : spires_inversion.interpolator.LutInterpolator, optional
        Pre-configured interpolator. If provided, overrides individual LUT parameters.
    lut_dataarray : xarray.DataArray, optional
        Not currently used. Reserved for future xarray support.
    max_eval : int, optional
        Maximum number of optimization iterations. Default is 100.
    x0 : array-like, optional
        Initial guess for [fsca, fshade, dust_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).
        1 = LN_COBYLA (constrained, derivative-free),
        2 = LN_NELDERMEAD (unconstrained simplex; ignores box bounds),
        3 = LD_SLSQP (gradient-based; degraded — uses NLopt's finite-diff fallback),
        4 = LN_NELDERMEAD on softmax-reparameterized cost (full softmax),
        5 = LN_BOBYQA on softmax-reparameterized cost (quadratic-model variant),
        6 = LN_NELDERMEAD on hybrid: softmax for fractions, clip-on-entry for
            dust/grain (recommended for real imagery).
        Algorithms 4-6 absorb the simplex (f_sca + f_shade + f_bg = 1, all ≥ 0)
        into the parameter transformation, so unconstrained NLopt solvers can
        replace COBYLA. On real imagery the hybrid (algorithm 6) is the
        recommended default replacement: it beats COBYLA on both fit quality
        and speed (~2.6× faster) and stays stable as `max_eval` is raised.

        Note: algorithms 4 and 5 (full softmax) suffer grain-bound saturation
        at high `max_eval` — the sigmoid reparameterization on dust/grain is
        asymptotically flat near the LUT bounds, letting the optimizer drift
        toward the upper bound while still lowering the residual. Algorithm 6
        does not have this problem because dust/grain stay in physical units
        with a clip in the objective, turning the bound into a true wall. See
        the "Softmax algorithms and grain-bound saturation" note in the README.

    Returns
    -------
    tuple
        Optimization results as (fsca, fshade, dust_concentration, grain_size) where:

        - fsca : float - Fractional snow-covered area (0-1)
        - fshade : float - Fractional shaded area (0-1)
        - dust_concentration : float - Dust concentration in snow (ppm)
        - grain_size : float - Effective snow grain radius (μm)

    Examples
    --------
    >>> import spires_inversion
    >>> import numpy as np
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> spectrum_background = np.array([0.0182,0.0265,0.0283,0.056067,0.095432,0.12036866,0.12491679,0.07888655,0.1406])
    >>> solar_angle = 55.73733298
    >>> interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> spires_inversion.speedy_invert(spectrum_target=spectrum_target, spectrum_background=spectrum_background,
    ...                      solar_angle=solar_angle, interpolator=interpolator, algorithm=1)
    (0.4089303296055291, 0.155201675059351, 138.79357872804923, 364.58404302094834)
    """

    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectrum_target)

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        dust_concentrations = interpolator.dust_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    return spires_inversion.core.invert(spectrum_background=spectrum_background, spectrum_target=spectrum_target,
                              spectrum_shade=spectrum_shade,
                              solar_angle=solar_angle, lut_bands=bands, lut_solar_angles=solar_angles,
                              lut_dust_concentrations=dust_concentrations, lut_grain_sizes=grain_sizes, lut_reflectances=reflectances,
                              max_eval=max_eval, x0=x0, algorithm=algorithm)


def speedy_invert_array1d(spectra_targets, spectra_backgrounds, obs_solar_angles, spectrum_shade=None,
                          bands=None, solar_angles=None, dust_concentrations=None, grain_sizes=None, reflectances=None,
                          interpolator=None, lut_dataarray=None, max_eval=100,
                          x0=np.array([0.5, 0.05, 10, 250]), algorithm=2):
    """
    Batch inversion of snow reflectance spectra for 1D arrays of observations.

    Efficiently processes multiple pixels/observations sequentially using optimized
    C++ implementations for improved performance.

    Parameters
    ----------
    spectra_targets : numpy.ndarray
        2D array of mixed spectra to invert with shape (n_observations, n_bands).
        Must have same length as `spectra_backgrounds` along first dimension.
    spectra_backgrounds : numpy.ndarray
        2D array of background (snow-free, R_0) spectra with shape (n_observations, n_bands).
        Must have same length as `spectra_targets` along first dimension.
    obs_solar_angles : numpy.ndarray
        1D array of solar zenith angles (degrees) for each observation.
        Must have same length as first dimension of `spectra_targets`.
    spectrum_shade : numpy.ndarray, optional
        1D array representing the ideal shaded spectrum for all observations.
        Must have same length as number of bands. If None, uses zeros (default: None).
    bands : numpy.ndarray, optional
        Band wavelength coordinates of reflectances. Required if interpolator not provided.
    solar_angles : numpy.ndarray, optional
        Solar angle coordinates of reflectances. Required if interpolator not provided.
    dust_concentrations : numpy.ndarray, optional
        Dust concentration coordinates of reflectances (ppm). Required if interpolator not provided.
    grain_sizes : numpy.ndarray, optional
        Grain size coordinates of reflectances (μm). Required if interpolator not provided.
    reflectances : numpy.ndarray, optional
        4D snow reflectance lookup table with dimensions (bands, solar_angles,
        dust_concentrations, grain_sizes). Required if interpolator not provided.
    interpolator : spires_inversion.interpolator.LutInterpolator, optional
        Pre-configured interpolator. If provided, overrides individual LUT parameters.
    lut_dataarray : xarray.DataArray, optional
        Not currently used. Reserved for future xarray support.
    max_eval : int, optional
        Maximum number of optimization iterations per observation (default: 100).
    x0 : array-like, optional
        Initial guess for [fsca, fshade, dust_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).
        1 = LN_COBYLA (constrained, derivative-free),
        2 = LN_NELDERMEAD (unconstrained simplex; ignores box bounds),
        3 = LD_SLSQP (gradient-based; degraded — uses NLopt's finite-diff fallback),
        4 = LN_NELDERMEAD on softmax-reparameterized cost (full softmax),
        5 = LN_BOBYQA on softmax-reparameterized cost (quadratic-model variant),
        6 = LN_NELDERMEAD on hybrid: softmax for fractions, clip-on-entry for
            dust/grain (recommended for real imagery).
        Algorithms 4-6 absorb the simplex (f_sca + f_shade + f_bg = 1, all ≥ 0)
        into the parameter transformation, so unconstrained NLopt solvers can
        replace COBYLA. On real imagery the hybrid (algorithm 6) is the
        recommended default replacement: it beats COBYLA on both fit quality
        and speed (~2.6× faster) and stays stable as `max_eval` is raised.

        Note: algorithms 4 and 5 (full softmax) suffer grain-bound saturation
        at high `max_eval` — the sigmoid reparameterization on dust/grain is
        asymptotically flat near the LUT bounds, letting the optimizer drift
        toward the upper bound while still lowering the residual. Algorithm 6
        does not have this problem because dust/grain stay in physical units
        with a clip in the objective, turning the bound into a true wall. See
        the "Softmax algorithms and grain-bound saturation" note in the README.

    Returns
    -------
    numpy.ndarray
        2D array of shape (n_observations, 4) containing inversion results:
        - results[:, 0] : Fractional snow-covered area (0-1)
        - results[:, 1] : Fractional shaded area (0-1)
        - results[:, 2] : Dust concentration in snow (ppm)
        - results[:, 3] : Effective snow grain radius (μm)

    Examples
    --------
    >>> import spires_inversion
    >>> import numpy as np
    >>> spectra_targets = np.array([[0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.0704336,0.06267947,0.3792],
    ...                            [0.2866,0.3046,0.324,0.34468558,0.35373732,0.35651454,0.1807259,0.16601688,0.3488]])
    >>> spectra_backgrounds = np.array([[0.0182,0.0265,0.0283,0.0560674,0.0954323,0.1203686,0.1249167,0.0788865,0.1406],
    ...                                [0.1002,0.1492,0.2088,0.2179780,0.2314920,0.2514020,0.3103066,0.2875081,0.2546]])
    >>> obs_solar_angles = np.array([55.73733298, 55.83733298])
    >>> interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> spires_inversion.speedy_invert_array1d(spectra_targets=spectra_targets, spectra_backgrounds=spectra_backgrounds,
    ...                            obs_solar_angles=obs_solar_angles, interpolator=interpolator, algorithm=1)
    array([[4.06627881e-01, 1.45134251e-01, 1.37503982e+02, 3.61158500e+02],
           [2.63873228e-01, 1.83226478e-01, 1.94343159e+02, 3.80170927e+02]])
    """
    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectra_targets[0])

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        dust_concentrations = interpolator.dust_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    n = spectra_targets.shape[0]
    results = np.empty((n, 4), dtype=np.double)

    spires_inversion.core.invert_array1d(spectra_targets=spectra_targets, spectra_backgrounds=spectra_backgrounds,
                               spectrum_shade=spectrum_shade,
                               obs_solar_angles=obs_solar_angles, lut_bands=bands, lut_solar_angles=solar_angles,
                               lut_dust_concentrations=dust_concentrations,
                               lut_grain_sizes=grain_sizes, lut_reflectances=reflectances, results=results,
                               max_eval=max_eval, x0=x0, algorithm=algorithm)
    return results


def _speedy_invert_grouped_block(
    spectra_targets,
    spectra_backgrounds,
    obs_solar_angles,
    *,
    spectrum_shade,
    bands,
    solar_angles,
    dust_concentrations,
    grain_sizes,
    reflectances,
    max_eval,
    x0,
    algorithm,
    valid_mask=None,
    grouping_method="group_mean",
    grouping_tolerance=0.02,
    grouping_reflectance_tol=None,
    grouping_background_tol=None,
    grouping_solar_zenith_tol=None,
):
    grouped = group_spectra_block(
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
        valid_mask=valid_mask,
        representative_method=grouping_method,
        tolerance=grouping_tolerance,
        reflectance_tol=grouping_reflectance_tol,
        background_tol=grouping_background_tol,
        solar_zenith_tol=grouping_solar_zenith_tol,
    )
    if grouped.n_groups == 0:
        return np.full(grouped.original_shape[:-1] + (4,), np.nan, dtype=np.double)

    grouped_results = speedy_invert_array1d(
        spectra_targets=grouped.representative_targets,
        spectra_backgrounds=grouped.representative_backgrounds,
        obs_solar_angles=grouped.representative_solar_zenith,
        spectrum_shade=spectrum_shade,
        bands=bands,
        solar_angles=solar_angles,
        dust_concentrations=dust_concentrations,
        grain_sizes=grain_sizes,
        reflectances=reflectances,
        max_eval=max_eval,
        x0=x0,
        algorithm=algorithm,
    )
    return scatter_group_results_block(grouped, grouped_results, fill_value=np.nan)


def speedy_invert_array2d(spectra_targets, spectra_backgrounds, obs_solar_angles, max_eval=100, x0=np.array([0.5, 0.05, 10, 250]), algorithm=2,
                          bands=None, solar_angles=None, dust_concentrations=None, grain_sizes=None, reflectances=None, interpolator=None,
                          spectrum_shade=None, valid_mask=None, use_grouping=False, grouping_method="group_mean",
                          grouping_tolerance=0.02, grouping_reflectance_tol=None, grouping_background_tol=None,
                          grouping_solar_zenith_tol=None):
    """
    Batch inversion of snow reflectance spectra for 2D spatial arrays.

    Processes entire images or 2D grids of observations efficiently using optimized
    C++ implementations. Ideal for processing satellite imagery or gridded data.

    Parameters
    ----------
    spectra_targets : numpy.ndarray
        3D array of mixed spectra to invert with shape (ny, nx, n_bands):
        - dim 0: y spatial dimension
        - dim 1: x spatial dimension
        - dim 2: spectral bands (must match order of `spectra_backgrounds`)
    spectra_backgrounds : numpy.ndarray
        3D array of background (snow-free, R_0) spectra with shape (ny, nx, n_bands):
        - dim 0: y spatial dimension (must match `spectra_targets`)
        - dim 1: x spatial dimension (must match `spectra_targets`)
        - dim 2: spectral bands (must match order of `spectra_targets`)
    obs_solar_angles : numpy.ndarray
        2D array of solar zenith angles (degrees) with shape (ny, nx).
        One angle per spatial location.
    max_eval : int, optional
        Maximum number of optimization iterations per pixel (default: 100).
    x0 : array-like, optional
        Initial guess for [fsca, fshade, dust_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).
        1 = LN_COBYLA (constrained, derivative-free),
        2 = LN_NELDERMEAD (unconstrained simplex; ignores box bounds),
        3 = LD_SLSQP (gradient-based; degraded — uses NLopt's finite-diff fallback),
        4 = LN_NELDERMEAD on softmax-reparameterized cost (full softmax),
        5 = LN_BOBYQA on softmax-reparameterized cost (quadratic-model variant),
        6 = LN_NELDERMEAD on hybrid: softmax for fractions, clip-on-entry for
            dust/grain (recommended for real imagery).
        Algorithms 4-6 absorb the simplex (f_sca + f_shade + f_bg = 1, all ≥ 0)
        into the parameter transformation, so unconstrained NLopt solvers can
        replace COBYLA. On real imagery the hybrid (algorithm 6) is the
        recommended default replacement: it beats COBYLA on both fit quality
        and speed (~2.6× faster) and stays stable as `max_eval` is raised.

        Note: algorithms 4 and 5 (full softmax) suffer grain-bound saturation
        at high `max_eval` — the sigmoid reparameterization on dust/grain is
        asymptotically flat near the LUT bounds, letting the optimizer drift
        toward the upper bound while still lowering the residual. Algorithm 6
        does not have this problem because dust/grain stay in physical units
        with a clip in the objective, turning the bound into a true wall. See
        the "Softmax algorithms and grain-bound saturation" note in the README.
    bands : numpy.ndarray, optional
        Band wavelength coordinates of reflectances. Required if interpolator not provided.
    solar_angles : numpy.ndarray, optional
        Solar angle coordinates of reflectances. Required if interpolator not provided.
    dust_concentrations : numpy.ndarray, optional
        Dust concentration coordinates of reflectances (ppm). Required if interpolator not provided.
    grain_sizes : numpy.ndarray, optional
        Grain size coordinates of reflectances (μm). Required if interpolator not provided.
    reflectances : numpy.ndarray, optional
        4D snow reflectance lookup table with dimensions (bands, solar_angles,
        dust_concentrations, grain_sizes). Required if interpolator not provided.
    interpolator : spires_inversion.interpolator.LutInterpolator, optional
        Pre-configured interpolator. If provided, overrides individual LUT parameters.
    spectrum_shade : numpy.ndarray, optional
        1D array representing the ideal shaded spectrum for all pixels.
        Must have same length as number of bands. If None, uses zeros.
    valid_mask : numpy.ndarray, optional
        Boolean mask with shape (ny, nx). Only used when ``use_grouping=True``.
    use_grouping : bool, optional
        If True, invert representative grouped spectra and scatter results back.
    grouping_method : str, optional
        Representative selection method: ``"group_mean"`` or ``"first_pixel"``.

    Returns
    -------
    numpy.ndarray
        3D array of shape (ny, nx, 4) containing inversion results:
        - results[:, :, 0] : Fractional snow-covered area (0-1)
        - results[:, :, 1] : Fractional shaded area (0-1)
        - results[:, :, 2] : Dust concentration in snow (ppm)
        - results[:, :, 3] : Effective snow grain radius (μm)

    Notes
    -----
    The shade spectrum is automatically set to zeros for all pixels when
    ``spectrum_shade`` is not provided.
    """
    
    if spectrum_shade is None:
        spectrum_shade = np.zeros(spectra_targets.shape[-1], dtype=np.double)
    else:
        spectrum_shade = np.asarray(spectrum_shade, dtype=np.double)

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        dust_concentrations = interpolator.dust_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    if use_grouping:
        return _speedy_invert_grouped_block(
            spectra_targets=spectra_targets,
            spectra_backgrounds=spectra_backgrounds,
            obs_solar_angles=obs_solar_angles,
            spectrum_shade=spectrum_shade,
            bands=bands,
            solar_angles=solar_angles,
            dust_concentrations=dust_concentrations,
            grain_sizes=grain_sizes,
            reflectances=reflectances,
            max_eval=max_eval,
            x0=x0,
            algorithm=algorithm,
            valid_mask=valid_mask,
            grouping_method=grouping_method,
            grouping_tolerance=grouping_tolerance,
            grouping_reflectance_tol=grouping_reflectance_tol,
            grouping_background_tol=grouping_background_tol,
            grouping_solar_zenith_tol=grouping_solar_zenith_tol,
        )

    results = np.empty((spectra_targets.shape[0], spectra_targets.shape[1], 4), dtype=np.double)


    spires_inversion.core.invert_array2d(spectra_backgrounds=spectra_backgrounds,
                               spectra_targets=spectra_targets,
                               spectrum_shade=spectrum_shade,
                               obs_solar_angles=obs_solar_angles,
                               lut_bands=bands, lut_solar_angles=solar_angles, lut_dust_concentrations=dust_concentrations,
                               lut_grain_sizes=grain_sizes, lut_reflectances=reflectances,
                               results=results,
                               max_eval=max_eval,
                               x0=x0,
                               algorithm=algorithm)
    return results



def speedy_invert_xarray(spectra_targets, spectra_backgrounds, obs_solar_angles, lut_dataarray,
                          spectrum_shade=None, max_eval=100,
                          x0=np.array([0.5, 0.05, 10, 250]), algorithm=2):
    """
    Batch inversion of snow reflectance spectra using xarray DataArrays.

    Provides a high-level interface for processing geospatial data with coordinate
    information preserved.

    Inputs must already be in canonical form: spectra as ``(y, x, band)`` and
    solar angles as ``(y, x)``, float64, with a ``band`` coordinate. The inputs
    are checked against the ``spires_contract`` validators on entry (once per
    call, not per pixel) and then passed to the C++ kernel as-is — this function
    does not transpose or cast. A misshaped input raises a clear ``ContractError``
    here rather than producing a cryptic failure inside the C++ kernel.

    Parameters
    ----------
    spectra_targets : xarray.DataArray
        Mixed spectra to invert with dimensions (y, x, band).
    spectra_backgrounds : xarray.DataArray
        Background (snow-free, R_0) spectra with dimensions (y, x, band).
        Must have same spatial dimensions as `spectra_targets`.
    obs_solar_angles : xarray.DataArray
        Solar zenith angles (degrees) with dimensions (y, x).
        One angle per spatial location.
    lut_dataarray : xarray.DataArray
        Lookup table with dimensions (band, solar_angle, dust_concentration, grain_size).
        Coordinates are extracted and used for interpolation.
    spectrum_shade : numpy.ndarray, optional
        1D array representing the ideal shaded spectrum.
        Must have same length as number of bands. If None, uses zeros (default: None).
    max_eval : int, optional
        Maximum number of optimization iterations per pixel (default: 100).
    x0 : array-like, optional
        Initial guess for [fsca, fshade, dust_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).
        1 = LN_COBYLA (constrained, derivative-free),
        2 = LN_NELDERMEAD (unconstrained simplex; ignores box bounds),
        3 = LD_SLSQP (gradient-based; degraded — uses NLopt's finite-diff fallback),
        4 = LN_NELDERMEAD on softmax-reparameterized cost (full softmax),
        5 = LN_BOBYQA on softmax-reparameterized cost (quadratic-model variant),
        6 = LN_NELDERMEAD on hybrid: softmax for fractions, clip-on-entry for
            dust/grain (recommended for real imagery).
        Algorithms 4-6 absorb the simplex (f_sca + f_shade + f_bg = 1, all ≥ 0)
        into the parameter transformation, so unconstrained NLopt solvers can
        replace COBYLA. On real imagery the hybrid (algorithm 6) is the
        recommended default replacement: it beats COBYLA on both fit quality
        and speed (~2.6× faster) and stays stable as `max_eval` is raised.

        Note: algorithms 4 and 5 (full softmax) suffer grain-bound saturation
        at high `max_eval` — the sigmoid reparameterization on dust/grain is
        asymptotically flat near the LUT bounds, letting the optimizer drift
        toward the upper bound while still lowering the residual. Algorithm 6
        does not have this problem because dust/grain stay in physical units
        with a clip in the objective, turning the bound into a true wall. See
        the "Softmax algorithms and grain-bound saturation" note in the README.

    Returns
    -------
    numpy.ndarray
        3D array of shape (ny, nx, 4) containing inversion results:
        - results[:, :, 0] : Fractional snow-covered area (0-1)
        - results[:, :, 1] : Fractional shaded area (0-1)
        - results[:, :, 2] : Dust concentration in snow (ppm)
        - results[:, :, 3] : Effective snow grain radius (μm)

    Notes
    -----
    Currently returns a numpy array. Future versions will return an xarray.DataArray
    with appropriate coordinates and metadata (see TODO comment in code).

    Raises
    ------
    spires_contract.ContractError
        If an input violates the I/O->inversion contract (wrong dimension order,
        missing/extra dimension, wrong dtype, or missing ``band`` coordinate).
        Validation runs once per call, so a misshaped array fails with a clear
        Python error here instead of a cryptic failure inside the C++ kernel.
    """
    validate_target_spectra(spectra_targets)
    validate_background_spectra(spectra_backgrounds)
    validate_solar_angles(obs_solar_angles)

    if spectrum_shade is None:
        spectrum_shade = np.zeros(spectra_targets.band.size, dtype=np.double)
   
    lut_bands = lut_dataarray.band
    lut_solar_angles = lut_dataarray.solar_angle
    lut_dust_concentrations = lut_dataarray.dust_concentration
    lut_grain_sizes = lut_dataarray.grain_size
    lut_reflectances = lut_dataarray.transpose('band', 'solar_angle', 'dust_concentration', 'grain_size').values

    results = np.empty((spectra_targets.y.size, spectra_targets.x.size, 4), dtype=np.double)

    spires_inversion.core.invert_array2d(spectra_backgrounds=spectra_backgrounds,
                               spectra_targets=spectra_targets,
                               spectrum_shade=spectrum_shade,
                               obs_solar_angles=obs_solar_angles,
                               lut_bands=lut_bands,
                               lut_solar_angles=lut_solar_angles,
                               lut_dust_concentrations=lut_dust_concentrations,
                               lut_grain_sizes=lut_grain_sizes,
                               lut_reflectances=lut_reflectances,
                               results=results,
                               max_eval=max_eval,
                               x0=x0,
                               algorithm=algorithm)
    
    # TODO: bootstrap the returned xarray!
    return results


def snow_diff_4(x, spectrum_target, spectrum_background, solar_angle, interpolator, shade):
    r"""
    Calculate spectral difference for 4-parameter snow model.

    Computes the Euclidean distance between observed and modeled spectra using
    a 4-parameter linear mixing model with snow, shade, and background components.

    .. math::

       \begin{align}
        R_{model}   & = R_{pure snow}( \phi_{sun}, c_{dust}, s_{grain}) * f_{sca}  \\
                    & + R_{shade} * f_{shade} \\
                    & + R_{0} * (1 - f_{sca} - f_{shade})
        \end{align}

    Parameters
    ----------
    x : array-like
        Model parameters:
        - x[0] : f_sca - Fractional snow-covered area (0-1)
        - x[1] : f_shade - Fractional shaded area (0-1)
        - x[2] : dust_concentration - Dust concentration in snow (ppm)
        - x[3] : grain_size - Effective snow grain radius (μm)
    spectrum_target : numpy.ndarray
        The observed mixed spectrum to match.
    spectrum_background : numpy.ndarray
        The background (snow-free, R_0) spectrum.
    solar_angle : float
        Solar zenith angle of the observation (degrees).
    interpolator : spires_inversion.interpolator.LutInterpolator
        Callable object that returns modeled snow spectrum given
        solar_angle, dust_concentration, and grain_size.
    shade : numpy.ndarray
        Ideal shade endmember spectrum.

    Returns
    -------
    float
        Euclidean distance between modeled and target spectra.

    Notes
    -----
    If f_sca is within 2%, consider using 3-parameter solution (snow_diff_3)
    to avoid overfitting.

    Examples
    --------
    >>> import spires_inversion
    >>> import numpy as np
    >>> interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> f_sca = 0.482
    >>> f_shade = 0.065
    >>> dust_concentration = 1000  # ppm
    >>> grain_size = 220  # μm
    >>> solar_angle = 55.73733298
    >>> x = [f_sca, f_shade, dust_concentration, grain_size]
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> spectrum_background = np.array([0.0182,0.0265,0.0283,0.056067,0.095432,0.12036866,0.12491679,0.07888655,0.1406])
    >>> shade = np.array([0,0,0,0,0,0,0,0,0])
    >>> diff = spires_inversion.snow_diff_4(x=x, spectrum_target=spectrum_target, spectrum_background=spectrum_background,
    ...                    solar_angle=solar_angle, interpolator=interpolator, shade=shade)
    >>> diff
    0.08870043573321955
    """

    model_reflectances = interpolator.interpolate_all(solar_angle=solar_angle,
                                                      dust_concentration=x[2],
                                                      grain_size=x[3])
    model_reflectances = model_reflectances * x[0] + shade * x[1] + spectrum_background * (1 - x[0] - x[1])
    distance = np.linalg.norm(spectrum_target - model_reflectances)
    return distance


def snow_diff_3(x, spectrum_target, solar_angle, interpolator, shade):
    r"""
    Calculate spectral difference for 3-parameter snow model.

    Computes the Euclidean distance between observed and modeled spectra using
    a simplified 3-parameter model where shade fills the non-snow fraction.

    .. math::

        \begin{align}
        R_{model} & = R_{pure snow}( \phi_{sun}, c_{dust}, s_{grain}) * f_{sca} \\
                  & + R_{shade} * (1-f_{sca})
        \end{align}

    Parameters
    ----------
    x : array-like
        Model parameters (note: only first 3 are used):
        - x[0] : f_sca - Fractional snow-covered area (0-1)
        - x[1] : dust_concentration - Dust concentration in snow (ppm)
        - x[2] : grain_size - Effective snow grain radius (μm)
    spectrum_target : numpy.ndarray
        The observed mixed spectrum to match.
    solar_angle : float
        Solar zenith angle of the observation (degrees).
    interpolator : spires_inversion.interpolator.LutInterpolator
        Callable object that returns modeled snow spectrum given
        solar_angle, dust_concentration, and grain_size.
    shade : numpy.ndarray
        Ideal shade endmember spectrum.

    Returns
    -------
    float
        Euclidean distance between modeled and target spectra.

    Notes
    -----
    This 3-parameter model assumes the non-snow fraction is entirely shade
    (no background component). Use when f_sca is near 100% to avoid overfitting.

    Examples
    --------
    >>> import spires_inversion
    >>> import numpy as np
    >>> interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> f_sca = 0.482
    >>> dust_concentration = 1000  # ppm
    >>> grain_size = 220  # μm
    >>> solar_angle = 55.73733298
    >>> x = [f_sca, dust_concentration, grain_size]
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> shade = np.array([0,0,0,0,0,0,0,0,0])
    >>> spires_inversion.snow_diff_3(x=x, spectrum_target=spectrum_target,
    ...                    solar_angle=solar_angle, interpolator=interpolator, shade=shade)
    0.06984199561833446
    """

    model_reflectances = interpolator.interpolate_all(solar_angle=solar_angle,
                                                      dust_concentration=x[1],
                                                      grain_size=x[2])

    model_reflectances = model_reflectances * x[0] + shade * (1 - x[0])
    distance = np.linalg.norm(spectrum_target - model_reflectances)
    return distance


def _x_to_z(x, dust_min, dust_max, grain_min, grain_max, eps=1e-6):
    """Inverse of the softmax/sigmoid reparameterization. Used to seed z0 from a
    physical x0 = [f_sca, f_shade, dust, grain]."""
    f_sca = float(np.clip(x[0], eps, 1 - eps))
    f_shade = float(np.clip(x[1], eps, 1 - eps))
    f_bg = float(np.clip(1.0 - f_sca - f_shade, eps, 1 - eps))
    z_snow = np.log(f_sca / f_bg)
    z_shade = np.log(f_shade / f_bg)

    u_d = np.clip((x[2] - dust_min) / (dust_max - dust_min), eps, 1 - eps)
    u_g = np.clip((x[3] - grain_min) / (grain_max - grain_min), eps, 1 - eps)
    z_dust = np.log(u_d / (1 - u_d))
    z_grain = np.log(u_g / (1 - u_g))
    return np.array([z_snow, z_shade, z_dust, z_grain])


def snow_diff_softmax(z, spectrum_target, spectrum_background, solar_angle, interpolator, shade,
                      dust_min, dust_max, grain_min, grain_max):
    r"""
    Spectral-difference cost in unconstrained (softmax-reparameterized) coordinates.

    Maps an unconstrained vector ``z = [z_snow, z_shade, z_dust, z_grain]`` to physical
    parameters such that the simplex constraints (f_sca, f_shade, f_bg ≥ 0,
    f_sca + f_shade + f_bg = 1) and the box bounds on dust/grain are satisfied by
    construction. This lets unconstrained solvers (Nelder-Mead, L-BFGS-B, BFGS) replace
    COBYLA / SLSQP on the fractional sub-problem.

    Reparameterization
    ------------------
    Fractions via softmax with z_bg pinned to 0 (gauge fix):

    .. math::
        (f_{sca}, f_{shade}, f_{bg}) = \mathrm{softmax}(z_{snow}, z_{shade}, 0)

    Dust and grain via sigmoid-scaled-to-bounds:

    .. math::
        d = d_{min} + (d_{max} - d_{min})\,\sigma(z_{dust}),\quad
        g = g_{min} + (g_{max} - g_{min})\,\sigma(z_{grain})

    The cost itself is the same Euclidean distance as :func:`snow_diff_4`.
    """
    e = np.exp(np.array([z[0], z[1], 0.0]) - max(z[0], z[1], 0.0))
    f_sca, f_shade, f_bg = e / e.sum()

    dust = dust_min + (dust_max - dust_min) / (1.0 + np.exp(-z[2]))
    grain = grain_min + (grain_max - grain_min) / (1.0 + np.exp(-z[3]))

    model_reflectances = interpolator.interpolate_all(solar_angle=solar_angle,
                                                      dust_concentration=dust,
                                                      grain_size=grain)
    model_reflectances = model_reflectances * f_sca + shade * f_shade + spectrum_background * f_bg
    return np.linalg.norm(spectrum_target - model_reflectances)


def speedy_invert_scipy_softmax(interpolator: spires_inversion.interpolator.LutInterpolator,
                                spectrum_target, spectrum_background, solar_angle,
                                shade=None, scipy_options=None, method='Nelder-Mead', z0=None):
    """
    Unconstrained scipy inversion via softmax reparameterization.

    Drops the inequality constraint ``1 - f_sca - f_shade ≥ 0`` and the box bounds by
    optimizing in an unconstrained space (see :func:`snow_diff_softmax`). Returns the
    same ``(res, model_refl)`` shape as :func:`speedy_invert_scipy`, with
    ``res.x = [f_sca, f_shade, dust, grain]`` in physical units.

    Parameters
    ----------
    method : str, optional
        Any unconstrained scipy method ('Nelder-Mead', 'BFGS', 'L-BFGS-B', 'Powell').
        Default 'Nelder-Mead'.
    z0 : array-like, optional
        Initial guess in z-space (length 4). If None, defaults to zeros, which
        corresponds to f = (1/3, 1/3, 1/3) and dust/grain at the bounds midpoint.
    """
    if shade is None:
        shade = np.zeros_like(spectrum_target)
    if scipy_options is None:
        scipy_options = {'disp': False, 'maxiter': 1000}

    dust_min = float(interpolator.dust_concentrations.min())
    dust_max = float(interpolator.dust_concentrations.max())
    grain_min = float(interpolator.grain_sizes.min())
    grain_max = float(interpolator.grain_sizes.max())

    if z0 is None:
        # Default physical guess: f_sca=0.5, f_shade=0.05, dust=10, grain=250
        # Map to z-space: z_bg pinned to 0, sigmoid logit for bounded params.
        z0 = _x_to_z(np.array([0.5, 0.05, 10.0, 250.0]),
                    dust_min, dust_max, grain_min, grain_max)

    res = scipy.optimize.minimize(
        snow_diff_softmax, z0, method=method, options=scipy_options,
        args=(spectrum_target, spectrum_background, solar_angle, interpolator, shade,
              dust_min, dust_max, grain_min, grain_max),
    )

    z = res.x
    e = np.exp(np.array([z[0], z[1], 0.0]) - max(z[0], z[1], 0.0))
    f_sca, f_shade, _ = e / e.sum()
    dust = dust_min + (dust_max - dust_min) / (1.0 + np.exp(-z[2]))
    grain = grain_min + (grain_max - grain_min) / (1.0 + np.exp(-z[3]))
    res.x = np.array([f_sca, f_shade, dust, grain])

    model_refl = interpolator.interpolate_all(solar_angle=solar_angle,
                                              dust_concentration=dust, grain_size=grain)
    return res, model_refl




def speedy_invert_scipy(interpolator: spires_inversion.interpolator.LutInterpolator, spectrum_target, spectrum_background,
                        solar_angle, shade=None,
                        scipy_options=None, mode=3, method='SLSQP'):
    """
    Invert snow spectra using scipy.optimize.minimize.

    Alternative implementation using SciPy's optimization routines instead of NLopt.
    Provides compatibility with legacy code and additional solver options.

    Parameters
    ----------
    interpolator : spires_inversion.interpolator.LutInterpolator
        Interpolator object with:
        - Attributes: `bands`, `solar_angles`, `dust_concentrations`, `grain_sizes`
        - Method: `interpolate_all(solar_angle, dust_concentration, grain_size)`
    spectrum_target : numpy.ndarray
        Target spectrum to be inverted. Must be same shape as `spectrum_background`.
    spectrum_background : numpy.ndarray
        Background (snow-free, R_0) spectrum. Must be same shape as `spectrum_target`.
    solar_angle : float
        Solar zenith angle of observation (degrees).
        Must use same units as interpolator coordinates.
    shade : numpy.ndarray, optional
        Ideal shade endmember spectrum. Must be same shape as `spectrum_target`.
        If None, uses zeros (default: None).
    scipy_options : dict, optional
        SciPy solver options. Default:
        `{'disp': False, 'iprint': 100, 'maxiter': 1000, 'ftol': 1e-9}`
    mode : int, optional
        Number of parameters in model (default: 3).
        3 = Simplified model (f_sca, dust, grain_size).
        4 = Full model (f_sca, f_shade, dust, grain_size).
        Use mode=3 when f_sca is near 100% to avoid overfitting.
    method : str, optional
        SciPy optimization method (default: 'SLSQP').
        Common options: 'SLSQP', 'L-BFGS-B', 'TNC'.

    Returns
    -------
    tuple
        (res, model_refl) where:

        - res : scipy.optimize.OptimizeResult
          Optimization result object. res.x contains:
          [f_sca, f_shade, dust_concentration, grain_size]
        - model_refl : numpy.ndarray
          The optimized modeled reflectance spectrum.

    See Also
    --------
    scipy.optimize.OptimizeResult : Documentation of result object
    speedy_invert : NLopt-based implementation (faster)

    Examples
    --------
    >>> import spires_inversion
    >>> import numpy as np
    >>> interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> interpolator.make_scipy_interpolator_legacy()
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> spectrum_background = np.array([0.0182,0.0265,0.0283,0.056067,0.095432,0.12036866,0.12491679,0.07888655,0.1406])
    >>> solar_angle = 24.0
    >>> res, model_refl = spires_inversion.speedy_invert_scipy(interpolator=interpolator,
    ...                                              spectrum_target=spectrum_target,
    ...                                              spectrum_background=spectrum_background,
    ...                                              solar_angle=solar_angle,
    ...                                              mode=3, method='SLSQP')
    >>> res.x
    array([4.36429085e-01, 5.63570915e-01, 9.91000000e+02, 4.12331162e+01])
    """

    bounds_fsca = [0, 1]
    bounds_fshade = [0, 1]
    bounds_dust = [interpolator.dust_concentrations.min(), interpolator.dust_concentrations.max()]
    bounds_grain = [interpolator.grain_sizes.min(), interpolator.grain_sizes.max()]

    if scipy_options is None:
        scipy_options = {'disp': False, 'iprint': 100, 'maxiter': 1000, 'ftol': 1e-9}

    if shade is None:
        shade = np.zeros_like(spectrum_target)

    if mode == 4:
        bounds = np.array([bounds_fsca, bounds_fshade, bounds_dust, bounds_grain])

        # inequality: constraint is => 0
        constraints = {"type": "ineq", "fun": lambda x: 1 - x[0] + x[1]}

        # initial guesses for f_sca, f_shade, dust, & grain size
        x0 = np.array([0.5, 0.05, 10, 250])

        res = scipy.optimize.minimize(snow_diff_4,
                                      x0,
                                      options=scipy_options,
                                      bounds=bounds,
                                      method=method,
                                      constraints=constraints,
                                      args=(spectrum_target, spectrum_background, solar_angle, interpolator, shade))
    elif mode == 3:
        bounds = np.array([bounds_fsca, bounds_dust, bounds_grain])

        # initial guesses for f_sca, dust, & grain size
        x0 = np.array([0.5, 10, 250])

        res = scipy.optimize.minimize(snow_diff_3,
                                      x0,
                                      options=scipy_options,
                                      bounds=bounds,
                                      method=method,
                                      args=(spectrum_target, solar_angle, interpolator, shade)
                                      )
        # insert f_shade (x[1] as 1-f_sca
        res.x = np.insert(res.x, 1, 1 - res.x[0])
    else:
        raise ValueError('mode must be either 4 or 3')

    # Lookup modelled reflectances
    model_refl = interpolator.interpolate_all(solar_angle=solar_angle, dust_concentration=res.x[2], grain_size=res.x[3])

    return res, model_refl


def index_to_value(index, coords):
    """
    Convert normalized index to coordinate value.

    Linearly interpolates between coordinate values based on a normalized
    index in the range [0, 1].

    Parameters
    ----------
    index : float
        Normalized index value between 0 and 1.
    coords : numpy.ndarray
        Array of coordinate values to interpolate between.

    Returns
    -------
    float
        Interpolated coordinate value.

    Notes
    -----
    Used internally by speedy_invert_scipy_normalized to convert
    normalized optimization parameters back to physical units.
    """
    idx = index * coords.size
    l_idx = int(idx)
    r_idx = l_idx + 1
    diff = coords[r_idx] - coords[l_idx]
    dist = idx - l_idx
    return coords[l_idx] + dist * diff


def speedy_invert_scipy_normalized(interpolator: spires_inversion.interpolator.LutInterpolator,
                                   spectrum_target, spectrum_background, solar_angle, spectrum_shade=None,
                                   method='COBYLA'):
    """
    Invert snow spectra with normalized parameter space.

    Performs optimization with all parameters scaled to [0, 1] range to improve
    convergence for solvers like COBYLA that don't support parameter-specific
    step sizes.

    Parameters
    ----------
    interpolator : spires_inversion.interpolator.LutInterpolator
        Interpolator object with lookup table and coordinate arrays.
    spectrum_target : numpy.ndarray
        Target spectrum to be inverted.
    spectrum_background : numpy.ndarray
        Background (snow-free, R_0) spectrum. Must be same shape as `spectrum_target`.
    solar_angle : float
        Solar zenith angle of observation (degrees).
    spectrum_shade : numpy.ndarray, optional
        Ideal shade endmember spectrum. Must be same shape as `spectrum_target`.
        If None, uses zeros (default: None).
    method : str, optional
        SciPy optimization method (default: 'COBYLA').
        COBYLA is recommended as it handles the normalized space well.

    Returns
    -------
    tuple
        (res, model_refl) where:

        - res : scipy.optimize.OptimizeResult
          Optimization result with res.x containing:
          [f_sca, f_shade, dust_concentration, grain_size]
          (dust and grain_size are converted back to physical units)
        - model_refl : numpy.ndarray
          The optimized modeled reflectance spectrum.

    Notes
    -----
    This function internally normalizes dust_concentration and grain_size
    to [0, 1] for optimization, then converts back to physical units.
    This improves convergence for algorithms that assume similar scales
    across parameters.
    """
    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectrum_target)

    scipy_options = {'disp': False, 'rhobeg': 0.05, 'maxiter': 100, 'tol': 1e-4}

    bounds_fsca = [0, 1]
    bounds_fshade = [0, 1]
    bounds_dust = [0, 1]
    bounds_grain = [0, 1]
    bounds = np.array([bounds_fsca, bounds_fshade, bounds_dust, bounds_grain], dtype=float)
    x0 = np.array([0.5, 0.05, 0.01, 0.1])

    res = scipy.optimize.minimize(spires_inversion.core.spectrum_difference_scaled,
                                  x0,
                                  method=method,
                                  options=scipy_options,
                                  bounds=bounds,
                                  args=(spectrum_background,
                                        spectrum_target,
                                        spectrum_shade,
                                        solar_angle,
                                        interpolator.bands,
                                        interpolator.solar_angles,
                                        interpolator.dust_concentrations,
                                        interpolator.grain_sizes,
                                        interpolator.reflectances)
                                  )

    res.x[2] = index_to_value(res.x[2], interpolator.dust_concentrations)
    res.x[3] = index_to_value(res.x[3], interpolator.grain_sizes)

    model_refl = interpolator.interpolate_all(solar_angle=solar_angle, dust_concentration=res.x[2], grain_size=res.x[3])
    return res, model_refl
