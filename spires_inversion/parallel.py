"""Dask-parallel inversion of snow reflectance spectra."""
import numpy as np

from spires_inversion.invert import _speedy_invert_grouped_block, speedy_invert_array2d


_VARIABLE_ATTRS = {
    'fsca': {
        'long_name': 'Fractional Snow-Covered Area',
        'units': '1',
        'valid_range': [0, 1],
    },
    'fshade': {
        'long_name': 'Fractional Shaded Area',
        'units': '1',
        'valid_range': [0, 1],
    },
    'dust_concentration': {
        'long_name': 'Dust Concentration in Snow',
        'units': 'ppm',
        'valid_range': [0, 10000],
    },
    'grain_size': {
        'long_name': 'Effective Snow Grain Radius',
        'units': 'μm',
        'valid_range': [10, 2000],
    },
}


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
    """Broadcast the LUT to all workers as a dask array."""
    arr = dask.array.from_array(reflectances)
    scattered = client.scatter(dict(arr.dask), broadcast=True)
    return dask.array.Array(
        scattered,
        name=arr.name,
        chunks=arr.chunks,
        dtype=arr.dtype,
        meta=arr._meta,
        shape=arr.shape,
    )


def _make_invert_chunk(
    max_eval,
    x0,
    algorithm,
    spectrum_shade,
    valid_mask_present,
    use_grouping,
    grouping_scope,
    grouping_method,
    grouping_tolerance,
    grouping_reflectance_tol,
    grouping_background_tol,
    grouping_solar_zenith_tol,
):
    """Build the per-chunk inversion function passed to apply_ufunc.

    Handles both the no-time (3D) and with-time (4D) layouts that
    apply_ufunc may hand to a single chunk.
    """
    def _invert(spectra_targets, spectra_backgrounds, obs_solar_angles, *args):
        if valid_mask_present:
            valid_mask = args[0]
            args = args[1:]
        else:
            valid_mask = None
        bands, solar_angles, dust, grain, reflectances = args

        common = dict(
            spectrum_shade=spectrum_shade,
            bands=bands,
            solar_angles=solar_angles,
            dust_concentrations=dust,
            grain_sizes=grain,
            reflectances=reflectances,
            max_eval=max_eval,
            x0=x0,
            algorithm=algorithm,
            use_grouping=use_grouping,
            grouping_method=grouping_method,
            grouping_tolerance=grouping_tolerance,
            grouping_reflectance_tol=grouping_reflectance_tol,
            grouping_background_tol=grouping_background_tol,
            grouping_solar_zenith_tol=grouping_solar_zenith_tol,
        )
        if use_grouping and grouping_scope == "chunk":
            return _speedy_invert_grouped_block(
                spectra_targets=spectra_targets,
                spectra_backgrounds=spectra_backgrounds,
                obs_solar_angles=obs_solar_angles,
                spectrum_shade=spectrum_shade,
                bands=bands,
                solar_angles=solar_angles,
                dust_concentrations=dust,
                grain_sizes=grain,
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

        if spectra_targets.ndim == 4:
            n_time = spectra_targets.shape[0]
            results = np.empty((n_time,) + spectra_targets.shape[1:3] + (4,))
            bg_t = spectra_backgrounds.ndim == 4
            sa_t = obs_solar_angles.ndim == 3
            mask_t = valid_mask is not None and valid_mask.ndim == 3
            for t in range(n_time):
                results[t] = speedy_invert_array2d(
                    spectra_targets=spectra_targets[t],
                    spectra_backgrounds=spectra_backgrounds[t] if bg_t else spectra_backgrounds,
                    obs_solar_angles=obs_solar_angles[t] if sa_t else obs_solar_angles,
                    valid_mask=None if valid_mask is None else valid_mask[t] if mask_t else valid_mask,
                    **common,
                )
            return results
        return speedy_invert_array2d(
            spectra_targets=spectra_targets,
            spectra_backgrounds=spectra_backgrounds,
            obs_solar_angles=obs_solar_angles,
            valid_mask=valid_mask,
            **common,
        )
    return _invert


def _to_dataset(results, attrs=None):
    ds = results.to_dataset(dim='property').rename({
        0: 'fsca', 1: 'fshade', 2: 'dust_concentration', 3: 'grain_size',
    })
    for name, variable_attrs in _VARIABLE_ATTRS.items():
        ds[name].attrs = variable_attrs
    if attrs:
        ds.attrs.update(attrs)
    return ds


def encode_results(ds, fill_value=-1, fsca_scale=100, fshade_scale=100,
                   fraction_dtype=np.int8, concentration_dtype=np.int16):
    """
    Encode an inversion result Dataset for compact storage.

    Replaces NaNs with ``fill_value``, scales the fractional variables, and
    casts to integer dtypes. The original Dataset is unchanged; a new one is
    returned with ``_FillValue``, ``scale_factor``, and ``add_offset`` attrs
    set so CF-aware readers (xarray, GDAL) decode back to physical units.

    Parameters
    ----------
    ds : xarray.Dataset
        Output of ``speedy_invert_dask`` with variables ``fsca``, ``fshade``,
        ``dust_concentration``, ``grain_size`` in physical units (NaN for nodata).
    fill_value : int, optional
        Sentinel for NaN pixels (default: -1).
    fsca_scale, fshade_scale : float, optional
        Multiplier applied before integer cast (default: 100, i.e. percent).
    fraction_dtype : numpy dtype, optional
        Integer type for fsca/fshade (default: ``np.int8``).
    concentration_dtype : numpy dtype, optional
        Integer type for dust_concentration/grain_size (default: ``np.int16``).

    Returns
    -------
    xarray.Dataset
        Dataset with the same variable names, encoded for storage.
    """
    import xarray

    encoded = ds.copy()
    scales = {
        'fsca': (fsca_scale, fraction_dtype),
        'fshade': (fshade_scale, fraction_dtype),
        'dust_concentration': (1, concentration_dtype),
        'grain_size': (1, concentration_dtype),
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
                       client=None, scatter_lut=True, valid_mask=None,
                       use_grouping=False, grouping_scope="scene",
                       grouping_method="mean_of_pixels", grouping_tolerance=0.02,
                       grouping_reflectance_tol=None, grouping_background_tol=None,
                       grouping_solar_zenith_tol=None):
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
        1D array representing the ideal shaded spectrum. If None, uses zeros.
    max_eval : int, optional
        Maximum optimization iterations per pixel (default: 100).
    x0 : array-like, optional
        Initial guess: [fsca, fshade, dust_conc (ppm), grain_size (μm)].
    algorithm : int, optional
        NLopt algorithm code (see speedy_invert_array2d).
    client : dask.distributed.Client, optional
        Dask client. If None, uses the default client when one is active.
    scatter_lut : bool, optional
        Broadcast the LUT to all workers (default: True).
    valid_mask : xarray.DataArray, optional
        Boolean mask with dimensions matching ``obs_solar_angles`` or
        broadcastable to each chunk's sample dimensions.
    use_grouping : bool, optional
        If True, invert representative grouped spectra and scatter back.
    grouping_scope : {"scene", "chunk"}, optional
        ``"scene"`` groups each time slice independently. ``"chunk"`` groups
        across all non-band dimensions inside each Dask chunk.

    Notes
    -----
    The C++ inversion releases the Python GIL, so Dask clients with
    ``threads_per_worker > 1`` get real parallel speedup and share a single
    LUT copy per worker process — usually preferable to many single-threaded
    workers when memory or IPC is a concern.

    Returns
    -------
    xarray.Dataset
        Dataset with variables fsca, fshade, dust_concentration, grain_size,
        preserving input coordinates and dimensions.

    See Also
    --------
    speedy_invert_xarray : Non-parallel xarray version
    speedy_invert_array2d : Core 2D array inversion function
    """
    import xarray

    dask, Client = _import_dask()
    client = _resolve_client(client, Client)

    if grouping_scope not in {"scene", "chunk"}:
        raise ValueError("grouping_scope must be one of {'scene', 'chunk'}")
    if spectrum_shade is None:
        spectrum_shade = np.zeros(len(interpolator.bands), dtype=np.double)
    else:
        spectrum_shade = np.asarray(spectrum_shade, dtype=np.double)

    if scatter_lut and client is not None:
        reflectances = _scatter_lut(client, interpolator.reflectances, dask)
    else:
        reflectances = interpolator.reflectances

    invert_chunk = _make_invert_chunk(
        max_eval=max_eval,
        x0=x0,
        algorithm=algorithm,
        spectrum_shade=spectrum_shade,
        valid_mask_present=valid_mask is not None,
        use_grouping=use_grouping,
        grouping_scope=grouping_scope,
        grouping_method=grouping_method,
        grouping_tolerance=grouping_tolerance,
        grouping_reflectance_tol=grouping_reflectance_tol,
        grouping_background_tol=grouping_background_tol,
        grouping_solar_zenith_tol=grouping_solar_zenith_tol,
    )

    ufunc_args = [
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
    ]
    input_core_dims = [
        ['band'], ['band'], [],
    ]
    if valid_mask is not None:
        ufunc_args.append(valid_mask)
        input_core_dims.append([])
    ufunc_args.extend([
        interpolator.bands,
        interpolator.solar_angles,
        interpolator.dust_concentrations,
        interpolator.grain_sizes,
        reflectances,
    ])
    input_core_dims.extend([
        ['bands'], ['sz'], ['dust'], ['grain'],
        ['bands', 'sz', 'dust', 'grain'],
    ])

    results = xarray.apply_ufunc(
        invert_chunk,
        *ufunc_args,
        dask='parallelized',
        input_core_dims=input_core_dims,
        output_core_dims=[['property']],
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={
            'allow_rechunk': False,
            'output_sizes': {'property': 4},
        },
        vectorize=False,
    )

    return _to_dataset(
        results,
        attrs={
            'grouping_enabled': bool(use_grouping),
            'grouping_scope': grouping_scope if use_grouping else 'none',
            'grouping_method': grouping_method if use_grouping else 'none',
            'grouping_tolerance': grouping_tolerance,
            'grouping_reflectance_tol': grouping_reflectance_tol,
            'grouping_background_tol': grouping_background_tol,
            'grouping_solar_zenith_tol': grouping_solar_zenith_tol,
        },
    )
