import spires_inversion.interpolator
import spires_inversion.core
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from pathlib import Path

from spires_contract.spectra import (
    validate_target_spectra,
    validate_background_spectra,
    validate_solar_angles,
)
from spires_contract.lut import validate_lut
from spires_contract import (
    ContractError,
    SpiresData,
    clusters_present,
    validate_for_inversion,
    validate_results,
)
from spires_contract import conventions as contract_conventions
from spires_inversion.lut import kernel_lut_arrays, load_reflectance_lut
from spires_inversion.results import build_results
import numpy as np
import scipy


DEFAULT_ALGORITHM = 6
DEFAULT_MAX_EVAL = 100
ALGORITHM_6_DEFAULT_MAX_EVAL = 200
DEFAULT_INITIAL_GRAIN_RADIUS_UM = 250.0
ALGORITHM_6_GRAIN_INITIAL_STEP_SQRT_UM = 4.0


def invert(
    data: SpiresData,
    *,
    lut,
    max_eval=None,
    algorithm=DEFAULT_ALGORITHM,
    initial_grain_radius_um=DEFAULT_INITIAL_GRAIN_RADIUS_UM,
    apply_valid_inversion_mask=True,
    n_workers=1,
):
    """Invert one contract-valid full-resolution or clustered scene.

    Clustered scenes are detected from the canonical cluster variables and
    inverted through their representative spectra. Both execution paths return
    canonical ``(y, x)`` results with ineligible pixels left as NaN.

    Parameters
    ----------
    data : spires_contract.SpiresData
        Prepared single scene. A complete canonical cluster schema selects
        clustered execution automatically.
    lut : xarray.Dataset or path-like
        Canonical NetCDF reflectance LUT or an already normalized in-memory
        Dataset.
    max_eval : int, optional
        Maximum objective evaluations per pixel or cluster representative.
        When omitted, Algorithm 6 uses 200; other algorithms retain 100.
    algorithm : int, optional
        Numerical-kernel algorithm code. Algorithm 6 is the object-path default.
    initial_grain_radius_um : float, optional
        Initial effective grain radius in micrometers. It must lie within the
        LUT grain-radius range. The default is 250.
    apply_valid_inversion_mask : bool, optional
        Apply ``valid_inversion_mask`` when it is present. Clustered inputs must
        have been formed using the same effective policy.
    n_workers : int, optional
        Number of in-process thread chunks used for row inversion. The default
        of one avoids nested parallelism; batch or Dask workers should normally
        retain that default.

    Returns
    -------
    spires_contract.SpiresData
        Replacement object carrying canonical float32 inversion results.
    """
    validate_for_inversion(data)
    max_eval = _resolve_max_eval(max_eval, algorithm)
    if not isinstance(apply_valid_inversion_mask, (bool, np.bool_)):
        raise TypeError("apply_valid_inversion_mask must be boolean")
    if (
        not isinstance(n_workers, (int, np.integer))
        or isinstance(n_workers, (bool, np.bool_))
        or n_workers < 1
    ):
        raise ValueError("n_workers must be a positive integer")
    n_workers = int(n_workers)

    reflectance_lut = load_reflectance_lut(lut, expected_lap_type="dust")
    _validate_scene_lut_bands(data.scene["reflectance"], reflectance_lut)
    lut_arrays = kernel_lut_arrays(reflectance_lut)
    initial_grain_radius_um = _validate_initial_grain_radius(
        initial_grain_radius_um,
        lut_arrays["sqrt_grain_radii"],
    )

    target = data.scene["reflectance"]
    valid_mask = data.scene.get("valid_inversion_mask")
    mask_applied = bool(apply_valid_inversion_mask and valid_mask is not None)

    if clusters_present(data):
        label = data.scene[contract_conventions.CLUSTER_LABEL_VARIABLE]
        clustered_mask_policy = bool(
            label.attrs[contract_conventions.CLUSTER_MASK_POLICY_ATTR]
        )
        if clustered_mask_policy != mask_applied:
            raise ContractError(
                "clustered input was built with "
                f"valid_inversion_mask_applied={clustered_mask_policy}, but "
                f"inversion requires {mask_applied}; recluster the scene with "
                "the requested mask policy"
            )
        kernel_results, eligible = _invert_clustered_scene(
            data,
            lut_arrays=lut_arrays,
            max_eval=max_eval,
            algorithm=algorithm,
            initial_grain_radius_um=initial_grain_radius_um,
            n_workers=n_workers,
        )
    else:
        kernel_results, eligible = _invert_full_resolution_scene(
            data,
            lut_arrays=lut_arrays,
            max_eval=max_eval,
            algorithm=algorithm,
            initial_grain_radius_um=initial_grain_radius_um,
            mask_applied=mask_applied,
            n_workers=n_workers,
        )

    eligibility_mask = target.isel(band=0, drop=True).copy(
        data=np.asarray(eligible, dtype=bool)
    )
    eligibility_mask.name = "effective_inversion_eligibility"

    results = build_results(kernel_results, eligibility_mask, lap_type="dust")
    results.attrs.update(
        _result_provenance(
            data,
            lut_source=lut,
            reflectance_lut=reflectance_lut,
            algorithm=algorithm,
            max_eval=max_eval,
            initial_grain_radius_um=initial_grain_radius_um,
            mask_applied=mask_applied,
            clustered=clusters_present(data),
        )
    )
    validate_results(
        results,
        scene=data.scene,
        eligibility_mask=eligibility_mask,
    )
    return data.assign_results(results)


def _resolve_max_eval(max_eval, algorithm):
    if max_eval is not None:
        return max_eval
    if algorithm == 6:
        return ALGORITHM_6_DEFAULT_MAX_EVAL
    return DEFAULT_MAX_EVAL


def _result_provenance(
    data,
    *,
    lut_source,
    reflectance_lut,
    algorithm,
    max_eval,
    initial_grain_radius_um,
    mask_applied,
    clustered,
):
    valid_mask = data.scene.get(contract_conventions.VALID_INVERSION_MASK_VARIABLE)
    identity, normalization = _lut_provenance(lut_source, reflectance_lut)
    attrs = {
        "spires_contract_version": _distribution_version("spires-contract"),
        "spires_inversion_version": _distribution_version("spires-inversion"),
        "inversion_algorithm": int(algorithm),
        "inversion_max_eval": int(max_eval),
        "inversion_initial_grain_radius_um": float(initial_grain_radius_um),
        "valid_inversion_mask_available": int(valid_mask is not None),
        "valid_inversion_mask_applied": int(mask_applied),
        "valid_inversion_mask_source": (
            "scene.valid_inversion_mask" if valid_mask is not None else "absent"
        ),
        "clustered_inversion": int(clustered),
        "reflectance_lut_identity": identity,
        "reflectance_lut_normalization": normalization,
    }
    if algorithm == 6:
        attrs["inversion_grain_initial_step_sqrt_um"] = (
            ALGORITHM_6_GRAIN_INITIAL_STEP_SQRT_UM
        )
    if clustered:
        label_attrs = data.scene[
            contract_conventions.CLUSTER_LABEL_VARIABLE
        ].attrs
        attrs["clustering_features"] = label_attrs.get("features", "")
        attrs["clustering_representative_method"] = label_attrs.get(
            "representative_method",
            "",
        )
        for name, value in label_attrs.items():
            if name.endswith("_tol"):
                attrs[f"clustering_{name}"] = value
    return attrs


def _lut_provenance(source, dataset):
    if isinstance(source, (str, PathLike)):
        identity = Path(source).name
    else:
        encoded_source = dataset.encoding.get("source")
        identity = dataset.attrs.get("reflectance_lut_identity")
        if identity is None and encoded_source:
            identity = Path(encoded_source).name
        if identity is None:
            identity = "in_memory"
    normalization = dataset.attrs.get(
        "reflectance_lut_normalization",
        "canonical_netcdf",
    )
    return str(identity), str(normalization)


def _distribution_version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _invert_full_resolution_scene(
    data,
    *,
    lut_arrays,
    max_eval,
    algorithm,
    initial_grain_radius_um,
    mask_applied,
    n_workers,
):
    target = data.scene["reflectance"]
    background = data.background
    solar_zenith = data.scene["solar_zenith"]
    eligible = (
        np.isfinite(target.values).all(axis=-1)
        & np.isfinite(background.values).all(axis=-1)
        & np.isfinite(solar_zenith.values)
    )
    if mask_applied:
        eligible &= np.asarray(
            data.scene["valid_inversion_mask"].values,
            dtype=bool,
        )

    kernel_results = np.full(target.shape[:2] + (4,), np.nan, dtype=np.float64)
    if np.any(eligible):
        kernel_results[eligible] = _invert_rows(
            target.values[eligible],
            background.values[eligible],
            solar_zenith.values[eligible],
            lut_arrays=lut_arrays,
            max_eval=max_eval,
            algorithm=algorithm,
            initial_grain_radius_um=initial_grain_radius_um,
            n_workers=n_workers,
        )
    return kernel_results, eligible


def _invert_clustered_scene(
    data,
    *,
    lut_arrays,
    max_eval,
    algorithm,
    initial_grain_radius_um,
    n_workers,
):
    scene = data.scene
    label = np.asarray(
        scene[contract_conventions.CLUSTER_LABEL_VARIABLE].values
    )
    eligible = label >= 0
    kernel_results = np.full(label.shape + (4,), np.nan, dtype=np.float64)

    n_clusters = scene.sizes[contract_conventions.CLUSTER_DIM]
    if n_clusters:
        cluster_results = _invert_rows(
            scene["cluster_representative_reflectance"].values,
            scene["cluster_representative_background"].values,
            scene["cluster_representative_solar_zenith"].values,
            lut_arrays=lut_arrays,
            max_eval=max_eval,
            algorithm=algorithm,
            initial_grain_radius_um=initial_grain_radius_um,
            n_workers=n_workers,
        )
        kernel_results[eligible] = cluster_results[label[eligible]]
    return kernel_results, eligible


def _invert_rows(
    spectra_targets,
    spectra_backgrounds,
    obs_solar_angles,
    *,
    lut_arrays,
    max_eval,
    algorithm,
    initial_grain_radius_um,
    n_workers,
):
    """Invert nonempty rows serially or in balanced contiguous thread chunks."""
    initial = np.asarray(
        [
            0.5,
            0.05,
            _within_axis(10.0, lut_arrays["lap_concentrations"]),
            _within_axis(
                np.sqrt(initial_grain_radius_um),
                lut_arrays["sqrt_grain_radii"],
            ),
        ],
        dtype=np.float64,
    )
    n_rows = spectra_targets.shape[0]
    worker_count = min(n_workers, n_rows)
    if worker_count == 1:
        return _invert_row_chunk(
            spectra_targets,
            spectra_backgrounds,
            obs_solar_angles,
            lut_arrays=lut_arrays,
            max_eval=max_eval,
            algorithm=algorithm,
            initial=initial,
        )

    boundaries = np.linspace(0, n_rows, worker_count + 1, dtype=np.int64)
    chunks = [
        slice(int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(worker_count)
    ]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _invert_row_chunk,
                spectra_targets[chunk],
                spectra_backgrounds[chunk],
                obs_solar_angles[chunk],
                lut_arrays=lut_arrays,
                max_eval=max_eval,
                algorithm=algorithm,
                initial=initial,
            )
            for chunk in chunks
        ]
        return np.concatenate([future.result() for future in futures], axis=0)


def _invert_row_chunk(
    spectra_targets,
    spectra_backgrounds,
    obs_solar_angles,
    *,
    lut_arrays,
    max_eval,
    algorithm,
    initial,
):
    return speedy_invert_array1d(
        spectra_targets=np.ascontiguousarray(spectra_targets),
        spectra_backgrounds=np.ascontiguousarray(spectra_backgrounds),
        obs_solar_angles=np.ascontiguousarray(obs_solar_angles),
        bands=lut_arrays["bands"],
        solar_angles=lut_arrays["solar_angles"],
        lap_concentrations=lut_arrays["lap_concentrations"],
        grain_sizes=lut_arrays["sqrt_grain_radii"],
        reflectances=lut_arrays["reflectances"],
        max_eval=max_eval,
        x0=initial,
        algorithm=algorithm,
    )


def _within_axis(value, axis):
    return float(np.clip(value, np.asarray(axis)[0], np.asarray(axis)[-1]))


def _validate_initial_grain_radius(value, sqrt_grain_axis):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("initial_grain_radius_um must be a real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("initial_grain_radius_um must be finite and > 0")

    axis = np.asarray(sqrt_grain_axis, dtype=np.float64)
    lower = float(axis[0] ** 2)
    upper = float(axis[-1] ** 2)
    tolerance = np.finfo(np.float64).eps * max(abs(lower), abs(upper), 1.0) * 8
    if value < lower - tolerance or value > upper + tolerance:
        raise ValueError(
            "initial_grain_radius_um must lie within the LUT grain-radius "
            f"range [{lower:g}, {upper:g}]"
        )
    return float(np.clip(value, lower, upper))


def _validate_scene_lut_bands(target, lut):
    scene_bands = np.asarray(target["band"].values)
    lut_bands = np.asarray(lut["reflectance"]["band"].values)
    if not np.array_equal(scene_bands, lut_bands):
        raise ContractError(
            "scene and reflectance LUT band coordinates must match exactly and "
            "in the same order"
        )


def _require_float32(name, arr):
    """Return `arr` as a C-contiguous float32 array, or raise if it isn't already
    float32.

    Used for the imagery arrays (targets/backgrounds), which cross the
    io -> inversion contract boundary. That boundary is float32-only (see
    spires-contract): spires-io emits float32. We refuse to silently down-cast a
    float64 input — that would be a lossy float64 -> float32 -> double round-trip
    and would hide a producer that hasn't migrated to float32.
    """
    a = np.asarray(arr)
    if a.dtype != np.float32:
        raise TypeError(
            f"{name} must be float32 at the inversion boundary, got {a.dtype}. "
            "The batch inversion path is float32-only; cast the producer's output "
            "to float32 (spires-io already emits float32) rather than passing float64."
        )
    return np.ascontiguousarray(a, dtype=np.float32)


def _invert_array2d(*, spectra_backgrounds, spectra_targets, spectrum_shade,
                    obs_solar_angles, bands, solar_angles,
                    lap_concentrations, grain_sizes, reflectances,
                    results, max_eval, x0, algorithm):
    """Run the float32-storage 2D batch C++ kernel.

    The three big arrays (imagery targets/backgrounds + the LUT reflectances) are
    passed to the kernel as C-contiguous float32, halving their memory footprint.
    The kernel promotes each value to double at read time, so the
    interpolation/cost math and NLopt run in full double precision. Coordinate
    axes, shade, solar angles, and results are always double.

    Both imagery and LUT are *enforced* float32 (`_require_float32`): imagery
    crosses the io contract boundary, and the LUT is now stored float32 by
    `LutInterpolator` (and validated by `spires_contract.validate_lut`), so a
    float64 array in either signals an unmigrated producer and raises rather than
    being silently down-cast. This makes the batch path a single deterministic
    float32 kernel.
    """
    # Coordinate axes / shade / solar angles are small -> always double.
    bands_d = np.ascontiguousarray(bands, dtype=np.float64)
    solar_angles_d = np.ascontiguousarray(solar_angles, dtype=np.float64)
    lap_d = np.ascontiguousarray(lap_concentrations, dtype=np.float64)
    grain_d = np.ascontiguousarray(grain_sizes, dtype=np.float64)
    shade_d = np.ascontiguousarray(spectrum_shade, dtype=np.float64)
    solar_obs_d = np.ascontiguousarray(obs_solar_angles, dtype=np.float64)

    spires_inversion.core.invert_array2d(
        spectra_backgrounds=_require_float32("spectra_backgrounds", spectra_backgrounds),
        spectra_targets=_require_float32("spectra_targets", spectra_targets),
        spectrum_shade=shade_d,
        obs_solar_angles=solar_obs_d,
        lut_bands=bands_d, lut_solar_angles=solar_angles_d,
        lut_lap_concentrations=lap_d, lut_grain_sizes=grain_d,
        lut_reflectances=_require_float32("reflectances", reflectances),
        results=results, max_eval=max_eval, x0=x0, algorithm=algorithm)


def speedy_invert(spectrum_target, spectrum_background, solar_angle, spectrum_shade=None,
                  bands=None, solar_angles=None, lap_concentrations=None, grain_sizes=None, reflectances=None,
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
    lap_concentrations : numpy.ndarray, optional
        LAP concentration coordinates of reflectances (ppm). Required if interpolator not provided.
    grain_sizes : numpy.ndarray, optional
        Grain size coordinates of reflectances (μm). Required if interpolator not provided.
    reflectances : numpy.ndarray, optional
        4D snow reflectance lookup table with dimensions (bands, solar_angles,
        lap_concentrations, grain_sizes). Required if interpolator not provided.
    interpolator : spires_inversion.interpolator.LutInterpolator, optional
        Pre-configured interpolator. If provided, overrides individual LUT parameters.
    lut_dataarray : xarray.DataArray, optional
        Not currently used. Reserved for future xarray support.
    max_eval : int, optional
        Maximum number of optimization iterations. Default is 100.
    x0 : array-like, optional
        Initial guess for [fsnow, fshade, lap_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).

        - ``1``: LN_COBYLA (constrained, derivative-free).
        - ``2``: LN_NELDERMEAD (unconstrained simplex; ignores box bounds).
        - ``3``: LD_SLSQP (gradient-based; finite-difference fallback).
        - ``4``: LN_NELDERMEAD with full softmax reparameterization.
        - ``5``: LN_BOBYQA with full softmax reparameterization.
        - ``6``: LN_NELDERMEAD with softmax fractions and clip-on-entry for
          LAP/grain (recommended for real imagery).

        Algorithms 4-6 absorb the simplex (f_snow + f_shade + f_bg = 1, all ≥ 0)
        into the parameter transformation, so unconstrained NLopt solvers can
        replace COBYLA. On real imagery the hybrid (algorithm 6) is the
        recommended default replacement: it beats COBYLA on both fit quality
        and speed (~2.6× faster) and stays stable as `max_eval` is raised.

        Note: algorithms 4 and 5 (full softmax) suffer grain-bound saturation
        at high `max_eval` — the sigmoid reparameterization on LAP/grain is
        asymptotically flat near the LUT bounds, letting the optimizer drift
        toward the upper bound while still lowering the residual. Algorithm 6
        does not have this problem because LAP/grain stay in physical units
        with a clip in the objective, turning the bound into a true wall. See
        the "Softmax algorithms and grain-bound saturation" note in the README.

    Returns
    -------
    tuple
        Optimization results as (fsnow, fshade, lap_concentration, grain_size) where:

        - fsnow : float - Fractional snow-covered area (0-1)
        - fshade : float - Fractional shaded area (0-1)
        - lap_concentration : float - LAP concentration in snow (ppm)
        - grain_size : float - Effective snow grain radius (μm)

    Examples
    --------
    >>> import spires_inversion
    >>> import numpy as np
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> spectrum_background = np.array([0.0182,0.0265,0.0283,0.056067,0.095432,0.12036866,0.12491679,0.07888655,0.1406])
    >>> solar_angle = 55.73733298
    >>> interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> fsnow, fshade, lap, grain = spires_inversion.speedy_invert(
    ...     spectrum_target=spectrum_target, spectrum_background=spectrum_background,
    ...     solar_angle=solar_angle, interpolator=interpolator, algorithm=1)
    >>> 0 <= fsnow <= 1 and 0 <= fshade <= 1 and fsnow + fshade <= 1.000001
    True
    >>> bool(lap >= 0) and grain > 0
    True
    """

    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectrum_target)

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        lap_concentrations = interpolator.lap_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    # The single-pixel C++ kernel stores the imagery spectra + LUT as float32
    # (promoted to double at read time). Unlike the batch path — which enforces
    # float32 at the io contract boundary — this is an interactive API taking
    # user-supplied spectra, so we cast them to float32 for convenience. The LUT
    # is already float32 from the interpolator; cast defensively for the
    # coords-only-supplied path.
    return spires_inversion.core.invert(
        spectrum_background=np.ascontiguousarray(spectrum_background, dtype=np.float32),
        spectrum_target=np.ascontiguousarray(spectrum_target, dtype=np.float32),
        spectrum_shade=np.ascontiguousarray(spectrum_shade, dtype=np.float64),
        solar_angle=solar_angle, lut_bands=bands, lut_solar_angles=solar_angles,
        lut_lap_concentrations=lap_concentrations, lut_grain_sizes=grain_sizes,
        lut_reflectances=np.ascontiguousarray(reflectances, dtype=np.float32),
        max_eval=max_eval, x0=x0, algorithm=algorithm)


def speedy_invert_array1d(spectra_targets, spectra_backgrounds, obs_solar_angles, spectrum_shade=None,
                          bands=None, solar_angles=None, lap_concentrations=None, grain_sizes=None, reflectances=None,
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
    lap_concentrations : numpy.ndarray, optional
        LAP concentration coordinates of reflectances (ppm). Required if interpolator not provided.
    grain_sizes : numpy.ndarray, optional
        Grain size coordinates of reflectances (μm). Required if interpolator not provided.
    reflectances : numpy.ndarray, optional
        4D snow reflectance lookup table with dimensions (bands, solar_angles,
        lap_concentrations, grain_sizes). Required if interpolator not provided.
    interpolator : spires_inversion.interpolator.LutInterpolator, optional
        Pre-configured interpolator. If provided, overrides individual LUT parameters.
    lut_dataarray : xarray.DataArray, optional
        Not currently used. Reserved for future xarray support.
    max_eval : int, optional
        Maximum number of optimization iterations per observation (default: 100).
    x0 : array-like, optional
        Initial guess for [fsnow, fshade, lap_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).

        - ``1``: LN_COBYLA (constrained, derivative-free).
        - ``2``: LN_NELDERMEAD (unconstrained simplex; ignores box bounds).
        - ``3``: LD_SLSQP (gradient-based; finite-difference fallback).
        - ``4``: LN_NELDERMEAD with full softmax reparameterization.
        - ``5``: LN_BOBYQA with full softmax reparameterization.
        - ``6``: LN_NELDERMEAD with softmax fractions and clip-on-entry for
          LAP/grain (recommended for real imagery).

        Algorithms 4-6 absorb the simplex (f_snow + f_shade + f_bg = 1, all ≥ 0)
        into the parameter transformation, so unconstrained NLopt solvers can
        replace COBYLA. On real imagery the hybrid (algorithm 6) is the
        recommended default replacement: it beats COBYLA on both fit quality
        and speed (~2.6× faster) and stays stable as `max_eval` is raised.

        Note: algorithms 4 and 5 (full softmax) suffer grain-bound saturation
        at high `max_eval` — the sigmoid reparameterization on LAP/grain is
        asymptotically flat near the LUT bounds, letting the optimizer drift
        toward the upper bound while still lowering the residual. Algorithm 6
        does not have this problem because LAP/grain stay in physical units
        with a clip in the objective, turning the bound into a true wall. See
        the "Softmax algorithms and grain-bound saturation" note in the README.

    Returns
    -------
    numpy.ndarray
        2D array of shape (n_observations, 4) containing inversion results:
        - results[:, 0] : Fractional snow-covered area (0-1)
        - results[:, 1] : Fractional shaded area (0-1)
        - results[:, 2] : LAP concentration in snow (ppm)
        - results[:, 3] : Effective snow grain radius (μm)

    Examples
    --------
    >>> import spires_inversion
    >>> import numpy as np
    >>> spectra_targets = np.array([[0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.0704336,0.06267947,0.3792],
    ...                            [0.2866,0.3046,0.324,0.34468558,0.35373732,0.35651454,0.1807259,0.16601688,0.3488]],
    ...                            dtype=np.float32)
    >>> spectra_backgrounds = np.array([[0.0182,0.0265,0.0283,0.0560674,0.0954323,0.1203686,0.1249167,0.0788865,0.1406],
    ...                                [0.1002,0.1492,0.2088,0.2179780,0.2314920,0.2514020,0.3103066,0.2875081,0.2546]],
    ...                                dtype=np.float32)
    >>> obs_solar_angles = np.array([55.73733298, 55.83733298])
    >>> interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> results = spires_inversion.speedy_invert_array1d(spectra_targets=spectra_targets,
    ...     spectra_backgrounds=spectra_backgrounds, obs_solar_angles=obs_solar_angles,
    ...     interpolator=interpolator, algorithm=1)
    >>> results.shape
    (2, 4)
    >>> bool(np.all((results[:, 0] >= 0) & (results[:, 0] <= 1)))  # fsnow in [0, 1]
    True
    """
    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectra_targets[0])

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        lap_concentrations = interpolator.lap_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    n = spectra_targets.shape[0]
    results = np.empty((n, 4), dtype=np.double)

    # Same float32-storage boundary as the 2D batch path: imagery and LUT are
    # enforced float32 (io contract + float32 interpolator/validate_lut);
    # coords/shade/solar/results stay double. See _invert_array2d.
    spires_inversion.core.invert_array1d(
        spectra_targets=_require_float32("spectra_targets", spectra_targets),
        spectra_backgrounds=_require_float32("spectra_backgrounds", spectra_backgrounds),
        spectrum_shade=np.ascontiguousarray(spectrum_shade, dtype=np.float64),
        obs_solar_angles=np.ascontiguousarray(obs_solar_angles, dtype=np.float64),
        lut_bands=np.ascontiguousarray(bands, dtype=np.float64),
        lut_solar_angles=np.ascontiguousarray(solar_angles, dtype=np.float64),
        lut_lap_concentrations=np.ascontiguousarray(lap_concentrations, dtype=np.float64),
        lut_grain_sizes=np.ascontiguousarray(grain_sizes, dtype=np.float64),
        lut_reflectances=_require_float32("reflectances", reflectances),
        results=results, max_eval=max_eval, x0=x0, algorithm=algorithm)
    return results


def speedy_invert_array2d(spectra_targets, spectra_backgrounds, obs_solar_angles, max_eval=100, x0=np.array([0.5, 0.05, 10, 250]), algorithm=2,
                          bands=None, solar_angles=None, lap_concentrations=None, grain_sizes=None, reflectances=None, interpolator=None):
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
        Initial guess for [fsnow, fshade, lap_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).

        - ``1``: LN_COBYLA (constrained, derivative-free).
        - ``2``: LN_NELDERMEAD (unconstrained simplex; ignores box bounds).
        - ``3``: LD_SLSQP (gradient-based; finite-difference fallback).
        - ``4``: LN_NELDERMEAD with full softmax reparameterization.
        - ``5``: LN_BOBYQA with full softmax reparameterization.
        - ``6``: LN_NELDERMEAD with softmax fractions and clip-on-entry for
          LAP/grain (recommended for real imagery).

        Algorithms 4-6 absorb the simplex (f_snow + f_shade + f_bg = 1, all ≥ 0)
        into the parameter transformation, so unconstrained NLopt solvers can
        replace COBYLA. On real imagery the hybrid (algorithm 6) is the
        recommended default replacement: it beats COBYLA on both fit quality
        and speed (~2.6× faster) and stays stable as `max_eval` is raised.

        Note: algorithms 4 and 5 (full softmax) suffer grain-bound saturation
        at high `max_eval` — the sigmoid reparameterization on LAP/grain is
        asymptotically flat near the LUT bounds, letting the optimizer drift
        toward the upper bound while still lowering the residual. Algorithm 6
        does not have this problem because LAP/grain stay in physical units
        with a clip in the objective, turning the bound into a true wall. See
        the "Softmax algorithms and grain-bound saturation" note in the README.
    bands : numpy.ndarray, optional
        Band wavelength coordinates of reflectances. Required if interpolator not provided.
    solar_angles : numpy.ndarray, optional
        Solar angle coordinates of reflectances. Required if interpolator not provided.
    lap_concentrations : numpy.ndarray, optional
        LAP concentration coordinates of reflectances (ppm). Required if interpolator not provided.
    grain_sizes : numpy.ndarray, optional
        Grain size coordinates of reflectances (μm). Required if interpolator not provided.
    reflectances : numpy.ndarray, optional
        4D snow reflectance lookup table with dimensions (bands, solar_angles,
        lap_concentrations, grain_sizes). Required if interpolator not provided.
    interpolator : spires_inversion.interpolator.LutInterpolator, optional
        Pre-configured interpolator. If provided, overrides individual LUT parameters.

    Returns
    -------
    numpy.ndarray
        3D array of shape (ny, nx, 4) containing inversion results:
        - results[:, :, 0] : Fractional snow-covered area (0-1)
        - results[:, :, 1] : Fractional shaded area (0-1)
        - results[:, :, 2] : LAP concentration in snow (ppm)
        - results[:, :, 3] : Effective snow grain radius (μm)

    Notes
    -----
    The shade spectrum is automatically set to zeros for all pixels. Future versions
    may support spatially-varying shade spectra.
    """
    
    spectrum_shade = np.zeros(spectra_targets.shape[-1], dtype=np.double)

    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectra_targets[0])

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        lap_concentrations = interpolator.lap_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    results = np.empty((spectra_targets.shape[0], spectra_targets.shape[1], 4), dtype=np.double)

    _invert_array2d(spectra_backgrounds=spectra_backgrounds,
                    spectra_targets=spectra_targets,
                    spectrum_shade=spectrum_shade,
                    obs_solar_angles=obs_solar_angles,
                    bands=bands, solar_angles=solar_angles,
                    lap_concentrations=lap_concentrations,
                    grain_sizes=grain_sizes, reflectances=reflectances,
                    results=results, max_eval=max_eval, x0=x0, algorithm=algorithm)
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
        Lookup table with dimensions (band, solar_angle, lap_concentration, grain_size).
        Coordinates are extracted and used for interpolation.
    spectrum_shade : numpy.ndarray, optional
        1D array representing the ideal shaded spectrum.
        Must have same length as number of bands. If None, uses zeros (default: None).
    max_eval : int, optional
        Maximum number of optimization iterations per pixel (default: 100).
    x0 : array-like, optional
        Initial guess for [fsnow, fshade, lap_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).

        - ``1``: LN_COBYLA (constrained, derivative-free).
        - ``2``: LN_NELDERMEAD (unconstrained simplex; ignores box bounds).
        - ``3``: LD_SLSQP (gradient-based; finite-difference fallback).
        - ``4``: LN_NELDERMEAD with full softmax reparameterization.
        - ``5``: LN_BOBYQA with full softmax reparameterization.
        - ``6``: LN_NELDERMEAD with softmax fractions and clip-on-entry for
          LAP/grain (recommended for real imagery).

        Algorithms 4-6 absorb the simplex (f_snow + f_shade + f_bg = 1, all ≥ 0)
        into the parameter transformation, so unconstrained NLopt solvers can
        replace COBYLA. On real imagery the hybrid (algorithm 6) is the
        recommended default replacement: it beats COBYLA on both fit quality
        and speed (~2.6× faster) and stays stable as `max_eval` is raised.

        Note: algorithms 4 and 5 (full softmax) suffer grain-bound saturation
        at high `max_eval` — the sigmoid reparameterization on LAP/grain is
        asymptotically flat near the LUT bounds, letting the optimizer drift
        toward the upper bound while still lowering the residual. Algorithm 6
        does not have this problem because LAP/grain stay in physical units
        with a clip in the objective, turning the bound into a true wall. See
        the "Softmax algorithms and grain-bound saturation" note in the README.

    Returns
    -------
    numpy.ndarray
        3D array of shape (ny, nx, 4) containing inversion results:
        - results[:, :, 0] : Fractional snow-covered area (0-1)
        - results[:, :, 1] : Fractional shaded area (0-1)
        - results[:, :, 2] : LAP concentration in snow (ppm)
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

    # Normalize the LUT to canonical dim order, then validate against the LUT
    # contract (dtype float32, dims, coords). We validate post-transpose because
    # this function accepts any transposable order and canonicalizes it here; the
    # contract's dtype/coord checks are what guard the C++ boundary.
    lut_dataarray = lut_dataarray.transpose(
        'band', 'solar_angle', 'lap_concentration', 'grain_size')
    validate_lut(lut_dataarray)

    lut_bands = lut_dataarray.band.values
    lut_solar_angles = lut_dataarray.solar_angle.values
    lut_lap_concentrations = lut_dataarray.lap_concentration.values
    lut_grain_sizes = lut_dataarray.grain_size.values
    lut_reflectances = lut_dataarray.values

    results = np.empty((spectra_targets.y.size, spectra_targets.x.size, 4), dtype=np.double)

    _invert_array2d(spectra_backgrounds=np.asarray(spectra_backgrounds),
                    spectra_targets=np.asarray(spectra_targets),
                    spectrum_shade=spectrum_shade,
                    obs_solar_angles=np.asarray(obs_solar_angles),
                    bands=lut_bands, solar_angles=lut_solar_angles,
                    lap_concentrations=lut_lap_concentrations,
                    grain_sizes=lut_grain_sizes, reflectances=lut_reflectances,
                    results=results, max_eval=max_eval, x0=x0, algorithm=algorithm)

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
        - x[0] : f_snow - Fractional snow-covered area (0-1)
        - x[1] : f_shade - Fractional shaded area (0-1)
        - x[2] : lap_concentration - LAP concentration in snow (ppm)
        - x[3] : grain_size - Effective snow grain radius (μm)
    spectrum_target : numpy.ndarray
        The observed mixed spectrum to match.
    spectrum_background : numpy.ndarray
        The background (snow-free, R_0) spectrum.
    solar_angle : float
        Solar zenith angle of the observation (degrees).
    interpolator : spires_inversion.interpolator.LutInterpolator
        Callable object that returns modeled snow spectrum given
        solar_angle, lap_concentration, and grain_size.
    shade : numpy.ndarray
        Ideal shade endmember spectrum.

    Returns
    -------
    float
        Euclidean distance between modeled and target spectra.

    Notes
    -----
    If f_snow is within 2%, consider using 3-parameter solution (snow_diff_3)
    to avoid overfitting.

    Examples
    --------
    >>> import spires_inversion
    >>> import numpy as np
    >>> interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> f_snow = 0.482
    >>> f_shade = 0.065
    >>> lap_concentration = 100  # ppm (within the LUT LAP range 0-991)
    >>> grain_size = 220  # μm
    >>> solar_angle = 55.73733298
    >>> x = [f_snow, f_shade, lap_concentration, grain_size]
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> spectrum_background = np.array([0.0182,0.0265,0.0283,0.056067,0.095432,0.12036866,0.12491679,0.07888655,0.1406])
    >>> shade = np.array([0,0,0,0,0,0,0,0,0])
    >>> diff = spires_inversion.snow_diff_4(x=x, spectrum_target=spectrum_target, spectrum_background=spectrum_background,
    ...                    solar_angle=solar_angle, interpolator=interpolator, shade=shade)
    >>> round(float(diff), 6)
    0.220393
    """

    model_reflectances = interpolator.interpolate_all(solar_angle=solar_angle,
                                                      lap_concentration=x[2],
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
        - x[0] : f_snow - Fractional snow-covered area (0-1)
        - x[1] : lap_concentration - LAP concentration in snow (ppm)
        - x[2] : grain_size - Effective snow grain radius (μm)
    spectrum_target : numpy.ndarray
        The observed mixed spectrum to match.
    solar_angle : float
        Solar zenith angle of the observation (degrees).
    interpolator : spires_inversion.interpolator.LutInterpolator
        Callable object that returns modeled snow spectrum given
        solar_angle, lap_concentration, and grain_size.
    shade : numpy.ndarray
        Ideal shade endmember spectrum.

    Returns
    -------
    float
        Euclidean distance between modeled and target spectra.

    Notes
    -----
    This 3-parameter model assumes the non-snow fraction is entirely shade
    (no background component). Use when f_snow is near 100% to avoid overfitting.

    Examples
    --------
    >>> import spires_inversion
    >>> import numpy as np
    >>> interpolator = spires_inversion.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> f_snow = 0.482
    >>> lap_concentration = 100  # ppm (within the LUT LAP range 0-991)
    >>> grain_size = 220  # μm
    >>> solar_angle = 55.73733298
    >>> x = [f_snow, lap_concentration, grain_size]
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> shade = np.array([0,0,0,0,0,0,0,0,0])
    >>> round(float(spires_inversion.snow_diff_3(x=x, spectrum_target=spectrum_target,
    ...                    solar_angle=solar_angle, interpolator=interpolator, shade=shade)), 6)
    0.164737
    """

    model_reflectances = interpolator.interpolate_all(solar_angle=solar_angle,
                                                      lap_concentration=x[1],
                                                      grain_size=x[2])

    model_reflectances = model_reflectances * x[0] + shade * (1 - x[0])
    distance = np.linalg.norm(spectrum_target - model_reflectances)
    return distance


def _x_to_z(x, lap_min, lap_max, grain_min, grain_max, eps=1e-6):
    """Inverse of the softmax/sigmoid reparameterization. Used to seed z0 from a
    physical x0 = [f_snow, f_shade, lap, grain]."""
    f_snow = float(np.clip(x[0], eps, 1 - eps))
    f_shade = float(np.clip(x[1], eps, 1 - eps))
    f_bg = float(np.clip(1.0 - f_snow - f_shade, eps, 1 - eps))
    z_snow = np.log(f_snow / f_bg)
    z_shade = np.log(f_shade / f_bg)

    u_d = np.clip((x[2] - lap_min) / (lap_max - lap_min), eps, 1 - eps)
    u_g = np.clip((x[3] - grain_min) / (grain_max - grain_min), eps, 1 - eps)
    z_dust = np.log(u_d / (1 - u_d))
    z_grain = np.log(u_g / (1 - u_g))
    return np.array([z_snow, z_shade, z_dust, z_grain])


def snow_diff_softmax(z, spectrum_target, spectrum_background, solar_angle, interpolator, shade,
                      lap_min, lap_max, grain_min, grain_max):
    r"""
    Spectral-difference cost in unconstrained (softmax-reparameterized) coordinates.

    Maps an unconstrained vector ``z = [z_snow, z_shade, z_dust, z_grain]`` to physical
    parameters such that the simplex constraints (f_snow, f_shade, f_bg ≥ 0,
    f_snow + f_shade + f_bg = 1) and the box bounds on LAP/grain are satisfied by
    construction. This lets unconstrained solvers (Nelder-Mead, L-BFGS-B, BFGS) replace
    COBYLA / SLSQP on the fractional sub-problem.

    Reparameterization
    ------------------
    Fractions via softmax with z_bg pinned to 0 (gauge fix):

    .. math::
        (f_{sca}, f_{shade}, f_{bg}) = \mathrm{softmax}(z_{snow}, z_{shade}, 0)

    LAP and grain via sigmoid-scaled-to-bounds:

    .. math::
        d = d_{min} + (d_{max} - d_{min})\,\sigma(z_{dust}),\quad
        g = g_{min} + (g_{max} - g_{min})\,\sigma(z_{grain})

    The cost itself is the same Euclidean distance as :func:`snow_diff_4`.
    """
    e = np.exp(np.array([z[0], z[1], 0.0]) - max(z[0], z[1], 0.0))
    f_snow, f_shade, f_bg = e / e.sum()

    lap = lap_min + (lap_max - lap_min) / (1.0 + np.exp(-z[2]))
    grain = grain_min + (grain_max - grain_min) / (1.0 + np.exp(-z[3]))

    model_reflectances = interpolator.interpolate_all(solar_angle=solar_angle,
                                                      lap_concentration=lap,
                                                      grain_size=grain)
    model_reflectances = model_reflectances * f_snow + shade * f_shade + spectrum_background * f_bg
    return np.linalg.norm(spectrum_target - model_reflectances)


def speedy_invert_scipy_softmax(interpolator: spires_inversion.interpolator.LutInterpolator,
                                spectrum_target, spectrum_background, solar_angle,
                                shade=None, scipy_options=None, method='Nelder-Mead', z0=None):
    """
    Unconstrained scipy inversion via softmax reparameterization.

    Drops the inequality constraint ``1 - f_snow - f_shade ≥ 0`` and the box bounds by
    optimizing in an unconstrained space (see :func:`snow_diff_softmax`). Returns the
    same ``(res, model_refl)`` shape as :func:`speedy_invert_scipy`, with
    ``res.x = [f_snow, f_shade, lap, grain]`` in physical units.

    Parameters
    ----------
    method : str, optional
        Any unconstrained scipy method ('Nelder-Mead', 'BFGS', 'L-BFGS-B', 'Powell').
        Default 'Nelder-Mead'.
    z0 : array-like, optional
        Initial guess in z-space (length 4). If None, defaults to zeros, which
        corresponds to f = (1/3, 1/3, 1/3) and LAP/grain at the bounds midpoint.
    """
    if shade is None:
        shade = np.zeros_like(spectrum_target)
    if scipy_options is None:
        scipy_options = {'disp': False, 'maxiter': 1000}

    lap_min = float(interpolator.lap_concentrations.min())
    lap_max = float(interpolator.lap_concentrations.max())
    grain_min = float(interpolator.grain_sizes.min())
    grain_max = float(interpolator.grain_sizes.max())

    if z0 is None:
        # Default physical guess: f_snow=0.5, f_shade=0.05, lap=10, grain=250
        # Map to z-space: z_bg pinned to 0, sigmoid logit for bounded params.
        z0 = _x_to_z(np.array([0.5, 0.05, 10.0, 250.0]),
                    lap_min, lap_max, grain_min, grain_max)

    res = scipy.optimize.minimize(
        snow_diff_softmax, z0, method=method, options=scipy_options,
        args=(spectrum_target, spectrum_background, solar_angle, interpolator, shade,
              lap_min, lap_max, grain_min, grain_max),
    )

    z = res.x
    e = np.exp(np.array([z[0], z[1], 0.0]) - max(z[0], z[1], 0.0))
    f_snow, f_shade, _ = e / e.sum()
    lap = lap_min + (lap_max - lap_min) / (1.0 + np.exp(-z[2]))
    grain = grain_min + (grain_max - grain_min) / (1.0 + np.exp(-z[3]))
    res.x = np.array([f_snow, f_shade, lap, grain])

    model_refl = interpolator.interpolate_all(solar_angle=solar_angle,
                                              lap_concentration=lap, grain_size=grain)
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
        - Attributes: `bands`, `solar_angles`, `lap_concentrations`, `grain_sizes`
        - Method: `interpolate_all(solar_angle, lap_concentration, grain_size)`
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
        3 = Simplified model (f_snow, lap, grain_size).
        4 = Full model (f_snow, f_shade, lap, grain_size).
        Use mode=3 when f_snow is near 100% to avoid overfitting.
    method : str, optional
        SciPy optimization method (default: 'SLSQP').
        Common options: 'SLSQP', 'L-BFGS-B', 'TNC'.

    Returns
    -------
    tuple
        (res, model_refl) where:

        - res : scipy.optimize.OptimizeResult
          Optimization result object. res.x contains:
          [f_snow, f_shade, lap_concentration, grain_size]
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
    >>> res.x.shape
    (4,)
    >>> bool(0 <= res.x[0] <= 1)  # fsnow in [0, 1]
    True
    """

    bounds_fsnow = [0, 1]
    bounds_fshade = [0, 1]
    bounds_lap = [interpolator.lap_concentrations.min(), interpolator.lap_concentrations.max()]
    bounds_grain = [interpolator.grain_sizes.min(), interpolator.grain_sizes.max()]

    if scipy_options is None:
        scipy_options = {'disp': False, 'iprint': 100, 'maxiter': 1000, 'ftol': 1e-9}

    if shade is None:
        shade = np.zeros_like(spectrum_target)

    if mode == 4:
        bounds = np.array([bounds_fsnow, bounds_fshade, bounds_lap, bounds_grain])

        # inequality: constraint is => 0
        constraints = {"type": "ineq", "fun": lambda x: 1 - x[0] + x[1]}

        # initial guesses for f_snow, f_shade, lap, & grain size
        x0 = np.array([0.5, 0.05, 10, 250])

        res = scipy.optimize.minimize(snow_diff_4,
                                      x0,
                                      options=scipy_options,
                                      bounds=bounds,
                                      method=method,
                                      constraints=constraints,
                                      args=(spectrum_target, spectrum_background, solar_angle, interpolator, shade))
    elif mode == 3:
        bounds = np.array([bounds_fsnow, bounds_lap, bounds_grain])

        # initial guesses for f_snow, lap, & grain size
        x0 = np.array([0.5, 10, 250])

        res = scipy.optimize.minimize(snow_diff_3,
                                      x0,
                                      options=scipy_options,
                                      bounds=bounds,
                                      method=method,
                                      args=(spectrum_target, solar_angle, interpolator, shade)
                                      )
        # insert f_shade (x[1] as 1-f_snow
        res.x = np.insert(res.x, 1, 1 - res.x[0])
    else:
        raise ValueError('mode must be either 4 or 3')

    # Lookup modelled reflectances
    model_refl = interpolator.interpolate_all(solar_angle=solar_angle, lap_concentration=res.x[2], grain_size=res.x[3])

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
          [f_snow, f_shade, lap_concentration, grain_size]
          (lap and grain_size are converted back to physical units)
        - model_refl : numpy.ndarray
          The optimized modeled reflectance spectrum.

    Notes
    -----
    This function internally normalizes lap_concentration and grain_size
    to [0, 1] for optimization, then converts back to physical units.
    This improves convergence for algorithms that assume similar scales
    across parameters.
    """
    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectrum_target)

    scipy_options = {'disp': False, 'rhobeg': 0.05, 'maxiter': 100, 'tol': 1e-4}

    bounds_fsnow = [0, 1]
    bounds_fshade = [0, 1]
    bounds_lap = [0, 1]
    bounds_grain = [0, 1]
    bounds = np.array([bounds_fsnow, bounds_fshade, bounds_lap, bounds_grain], dtype=float)
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
                                        interpolator.lap_concentrations,
                                        interpolator.grain_sizes,
                                        interpolator.reflectances)
                                  )

    res.x[2] = index_to_value(res.x[2], interpolator.lap_concentrations)
    res.x[3] = index_to_value(res.x[3], interpolator.grain_sizes)

    model_refl = interpolator.interpolate_all(solar_angle=solar_angle, lap_concentration=res.x[2], grain_size=res.x[3])
    return res, model_refl
