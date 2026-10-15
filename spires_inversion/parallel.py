"""Dask-parallel inversion of snow reflectance spectra."""
import warnings

import numpy as np

from spires_contract import conventions as _c
from spires_contract import validate_results
from spires_inversion.invert import speedy_invert_array2d
from spires_inversion.results import build_results, validate_result_structure


def _import_dask():
    try:
        import dask
        import dask.array  # noqa: F401  (registers the array submodule)
        from dask.distributed import Client
    except ImportError as exc:
        raise ImportError(
            "Dask is required for parallel processing. "
            "Install with: conda install -c conda-forge dask distributed"
        ) from exc
    return dask, Client


def _resolve_client(client, Client):
    """Return the user-supplied client, or the active default if any."""
    if client is not None:
        return client
    try:
        return Client.current()
    except ValueError:
        return None


def _scatter_lut(client, reflectances, dask):
    """Broadcast the LUT to all workers as a single-chunk dask array.

    The LUT's four dims are *input core dims* of this module's ``apply_ufunc``
    call, so the array must be exactly one chunk; ``chunks=-1`` states that
    rather than inheriting it from ``dask.config``'s ``array.chunk-size``
    (128 MiB default), under which a larger LUT silently split into several
    chunks and failed with ``Dimension 'dim1' with different chunksize
    present`` from a code path that never mentions chunking.

    Scattering is done as a **list** with ``hash=False`` so the key is drawn
    from ``uuid.uuid4()``. Scattering a dict (or a list with the default
    ``hash=True``) yields content-addressed keys, identical on every call for
    identical LUT content, which is the precondition for dask's
    release-vs-registration race: call N's futures release a key over the
    client's batched comm while call N+1 re-registers it over a
    ``scheduler.update_data`` RPC, with nothing ordering the two. Losing that
    race raises ``FutureCancelledError: ... lost dependencies`` or hangs.

    The returned array owns the ``Future`` in its graph, so the data stays
    resident on the workers for as long as the array is referenced.
    """
    # from_array supplies chunks/dtype/meta/shape; its own graph is discarded
    # in favour of the single scattered future.
    arr = dask.array.from_array(reflectances, chunks=-1)
    (future,) = client.scatter([reflectances], broadcast=True, hash=False)
    name = "spires-lut-" + future.key
    return dask.array.Array(
        {(name,) + (0,) * reflectances.ndim: future},
        name=name,
        chunks=arr.chunks,
        dtype=arr.dtype,
        meta=arr._meta,
        shape=arr.shape,
    )


#: dtype of the blocks ``_make_invert_chunk``'s kernel returns. Must stay in
#: step with the ``output_dtypes`` declared on the ``apply_ufunc`` call below:
#: ``output_dtypes`` is only a *declaration*, so a mismatch yields a dask array
#: that reports one dtype and materializes another, and whose ``.astype`` to
#: the declared dtype is a silent no-op (dask returns ``self``). The solver
#: itself is unaffected — coordinate axes, shade, solar angles and the kernel's
#: own results buffer stay ``double``; this narrows only at the dask boundary,
#: which is also what makes the result bytes match the float32 results contract.
_RESULT_DTYPE = np.float32


def _make_invert_chunk(max_eval, x0, algorithm):
    """Build the per-chunk inversion function passed to apply_ufunc.

    Handles both the no-time (3D) and with-time (4D) layouts that
    apply_ufunc may hand to a single chunk. Blocks are returned as
    ``_RESULT_DTYPE`` to match the declared ``output_dtypes``.
    """
    def _invert(spectra_targets, spectra_backgrounds, obs_solar_angles,
                bands, solar_angles, lap, grain, reflectances):
        common = dict(
            bands=bands,
            solar_angles=solar_angles,
            lap_concentrations=lap,
            grain_sizes=grain,
            reflectances=reflectances,
            max_eval=max_eval,
            x0=x0,
            algorithm=algorithm,
        )
        if spectra_targets.ndim == 4:
            n_time = spectra_targets.shape[0]
            results = np.empty(
                (n_time,) + spectra_targets.shape[1:3] + (4,),
                dtype=_RESULT_DTYPE,
            )
            bg_t = spectra_backgrounds.ndim == 4
            sa_t = obs_solar_angles.ndim == 3
            for t in range(n_time):
                # speedy_invert_array2d returns double; assignment into the
                # float32 buffer narrows each scene as it lands.
                results[t] = speedy_invert_array2d(
                    spectra_targets=spectra_targets[t],
                    spectra_backgrounds=spectra_backgrounds[t] if bg_t else spectra_backgrounds,
                    obs_solar_angles=obs_solar_angles[t] if sa_t else obs_solar_angles,
                    **common,
                )
            return results
        return speedy_invert_array2d(
            spectra_targets=spectra_targets,
            spectra_backgrounds=spectra_backgrounds,
            obs_solar_angles=obs_solar_angles,
            **common,
        ).astype(_RESULT_DTYPE, copy=False)
    return _invert


def _to_dataset(results):
    return build_results(
        results,
        lap_type=_c.SUPPORTED_LAP_TYPES[0],
        grain_output="grain_radius",
    )


def encode_results(ds, fill_value=-1, fsnow_scale=100, fshade_scale=100,
                   fraction_dtype=np.int8, concentration_dtype=np.int16,
                   *, fsca_scale=None):
    """
    Encode an inversion result Dataset for compact storage.

    Replaces NaNs with ``fill_value``, scales the fractional variables, and
    casts to integer dtypes. The original Dataset is unchanged; a new one is
    returned with ``_FillValue``, ``scale_factor``, and ``add_offset`` attrs
    set so CF-aware readers (xarray, GDAL) decode back to physical units.

    Parameters
    ----------
    ds : xarray.Dataset
        Output of ``speedy_invert_dask`` with variables ``fsnow``, ``fshade``,
        ``lap_concentration``, ``grain_radius`` in physical units (NaN for nodata).
    fill_value : int, optional
        Sentinel for NaN pixels (default: -1).
    fsnow_scale, fshade_scale : float, optional
        Multiplier applied before integer cast (default: 100, i.e. percent).
    fraction_dtype : numpy dtype, optional
        Integer type for fsnow/fshade (default: ``np.int8``).
    concentration_dtype : numpy dtype, optional
        Integer type for lap_concentration/grain_radius (default: ``np.int16``).

    Returns
    -------
    xarray.Dataset
        Dataset with the same variable names, encoded for storage.

    Notes
    -----
    This is a storage-only representation, not a canonical in-memory result
    Dataset. ``fsca_scale`` is a deprecated alias for ``fsnow_scale``.
    """
    import xarray

    if fsca_scale is not None:
        if fsnow_scale != 100:
            raise TypeError("use only one of fsnow_scale or deprecated fsca_scale")
        warnings.warn(
            "fsca_scale is deprecated; use fsnow_scale",
            DeprecationWarning,
            stacklevel=2,
        )
        fsnow_scale = fsca_scale

    encoded = ds.copy()
    scales = {
        'fsnow': (fsnow_scale, fraction_dtype),
        'fshade': (fshade_scale, fraction_dtype),
        'lap_concentration': (1, concentration_dtype),
        'grain_radius': (1, concentration_dtype),
    }
    for name, (scale, dtype) in scales.items():
        if name not in encoded:
            continue
        var = encoded[name]
        scaled = var * scale if scale != 1 else var
        encoded[name] = xarray.where(np.isnan(var), fill_value, scaled).astype(dtype)
        encoded[name].attrs = dict(var.attrs)
        encoded[name].attrs['_FillValue'] = dtype(fill_value)
        if scale != 1:
            encoded[name].attrs['scale_factor'] = 1.0 / scale
            encoded[name].attrs['add_offset'] = 0.0
    return encoded


def speedy_invert_dask(spectra_targets, spectra_backgrounds, obs_solar_angles,
                       interpolator, spectrum_shade=None, max_eval=100,
                       x0=np.array([0.5, 0.05, 10, 250]), algorithm=2,
                       client=None, scatter_lut=True):
    """
    Parallel inversion of snow reflectance spectra using Dask and xarray.

    This method enables distributed processing of large satellite imagery datasets
    by leveraging Dask's parallel computation capabilities through xarray.apply_ufunc.
    It's particularly useful for processing time series of satellite imagery where
    the data is too large to fit in memory.

    Parameters
    ----------
    spectra_targets : xarray.DataArray
        Mixed spectra to invert. Must have a 'band' dimension and can have
        any combination of spatial (x, y) and temporal (time) dimensions.
        Shape: (time, y, x, band) or (y, x, band).
    spectra_backgrounds : xarray.DataArray
        Background (snow-free, R_0) spectra with same dimensions as targets
        except potentially missing the time dimension if using static backgrounds.
        Shape: (y, x, band).
    obs_solar_angles : xarray.DataArray
        Solar zenith angles (degrees) for each observation.
        Shape: (time, y, x) or (y, x).
    interpolator : spires_inversion.interpolator.LutInterpolator
        Lookup table interpolator object.
    spectrum_shade : numpy.ndarray, optional
        Currently unused: speedy_invert_array2d hardcodes a zero shade spectrum.
        Accepted for forward compatibility.
    max_eval : int, optional
        Maximum optimization iterations per pixel (default: 100).
    x0 : array-like, optional
        Initial guess: [fsnow, fshade, lap_conc (ppm), grain_size (μm)].
    algorithm : int, optional
        NLopt algorithm code (see speedy_invert_array2d).
    client : dask.distributed.Client, optional
        Dask client. If None, uses the default client when one is active.
    scatter_lut : bool, optional
        Broadcast the LUT to all workers (default: True).

    Notes
    -----
    The C++ inversion releases the Python GIL, so Dask clients with
    ``threads_per_worker > 1`` get real parallel speedup and share a single
    LUT copy per worker process — usually preferable to many single-threaded
    workers when memory or IPC is a concern.

    Returns
    -------
    xarray.Dataset
        Dataset with variables fsnow, fshade, lap_concentration, grain_radius,
        preserving input coordinates and dimensions.

    See Also
    --------
    speedy_invert_xarray : Non-parallel xarray version
    speedy_invert_array2d : Core 2D array inversion function
    """
    import xarray

    dask, Client = _import_dask()
    client = _resolve_client(client, Client)

    if scatter_lut and client is not None:
        reflectances = _scatter_lut(client, interpolator.reflectances, dask)
    else:
        reflectances = interpolator.reflectances

    invert_chunk = _make_invert_chunk(max_eval, x0, algorithm)

    results = xarray.apply_ufunc(
        invert_chunk,
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
        interpolator.bands,
        interpolator.solar_angles,
        interpolator.lap_concentrations,
        interpolator.grain_sizes,
        reflectances,
        dask='parallelized',
        input_core_dims=[
            ['band'], ['band'], [],
            ['bands'], ['sz'], ['lap'], ['grain'],
            ['bands', 'sz', 'lap', 'grain'],
        ],
        output_core_dims=[['property']],
        output_dtypes=[_RESULT_DTYPE],
        dask_gufunc_kwargs={
            'allow_rechunk': False,
            'output_sizes': {'property': 4},
        },
        vectorize=False,
    )

    ds = _to_dataset(results)

    # Every result receives lazy-safe structural validation. Full contract
    # validation reads values, so it remains limited to eager single scenes;
    # lazy or time-stacked callers validate each scene after materialization.
    validate_result_structure(ds)
    if _is_eager_single_scene(ds):
        validate_results(ds)

    return ds


def _is_eager_single_scene(ds):
    """True if ds is numpy-backed (not dask) and has exactly (y, x) dims."""
    if any(ds[name].chunks is not None for name in _c.RESULT_VARIABLES):
        return False
    return all(
        tuple(ds[name].dims) == _c.RESULT_DIMS for name in _c.RESULT_VARIABLES
    )
