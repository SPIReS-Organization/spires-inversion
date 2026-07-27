"""Reflectance-LUT loading and normalization for standardized inversion."""

from os import PathLike
from pathlib import Path

import numpy as np
import xarray as xr

from spires_contract import ContractError, validate_reflectance_lut


_REFLECTANCE_DIMS = (
    "band",
    "solar_angle",
    "lap_concentration",
    "sqrt_grain_radius",
)


def load_reflectance_lut(source, *, expected_lap_type="dust"):
    """Load and validate a canonical NetCDF reflectance LUT.

    ``source`` may be an in-memory :class:`xarray.Dataset` or a NetCDF path.
    Legacy MATLAB files must be passed through
    :func:`load_matlab_reflectance_lut` explicitly.
    """
    if isinstance(source, xr.Dataset):
        dataset = source
    elif isinstance(source, (str, PathLike)):
        with xr.open_dataset(source) as opened:
            dataset = opened.load()
    else:
        raise TypeError(
            "reflectance LUT must be an xarray.Dataset or NetCDF path, "
            f"got {type(source).__name__}"
        )

    validate_reflectance_lut(dataset, expected_lap_type=expected_lap_type)
    return dataset


def load_matlab_reflectance_lut(
    path,
    *,
    lap_type,
    lap_concentration_units,
    grain_radius_units,
    solar_angle_units,
):
    """Normalize a transitional MATLAB reflectance LUT to the v1 contract.

    Unit declarations are required because the legacy file layout does not
    carry reliable axis metadata. Supported LAP units are ``"ppm"`` and
    ``"fraction"``; the latter is converted to ppm. Grain radius must be
    declared in micrometres and solar angle in degrees.
    """
    # Local import avoids making the canonical NetCDF loader depend on the
    # transitional MATLAB reader.
    from spires_inversion.interpolator import LutInterpolator

    legacy = LutInterpolator(lut_file=path)
    lap_values = _lap_concentration_to_ppm(
        legacy.lap_concentrations, lap_concentration_units
    )
    grain_radius = _require_micrometres(
        legacy.grain_sizes, grain_radius_units
    )
    _require_degrees(solar_angle_units)

    if np.any(~np.isfinite(grain_radius)) or np.any(grain_radius < 0):
        raise ContractError(
            "legacy MATLAB grain-radius coordinates must be finite and nonnegative"
        )

    dataset = xr.Dataset(
        {
            "reflectance": xr.DataArray(
                np.ascontiguousarray(legacy.reflectances, dtype=np.float32),
                dims=_REFLECTANCE_DIMS,
                coords={
                    "band": legacy.bands,
                    "solar_angle": xr.DataArray(
                        np.asarray(legacy.solar_angles),
                        dims="solar_angle",
                        attrs={"units": "degrees"},
                    ),
                    "lap_concentration": xr.DataArray(
                        lap_values,
                        dims="lap_concentration",
                        attrs={"units": "ppm", "lap_type": lap_type},
                    ),
                    "sqrt_grain_radius": xr.DataArray(
                        np.sqrt(grain_radius),
                        dims="sqrt_grain_radius",
                        attrs={"units": "um^0.5"},
                    ),
                },
            )
        }
    )
    dataset.attrs.update(
        {
            "reflectance_lut_identity": Path(path).name,
            "reflectance_lut_normalization": "legacy_matlab_normalized",
        }
    )
    validate_reflectance_lut(dataset, expected_lap_type=lap_type)
    return dataset


def kernel_lut_arrays(dataset):
    """Return validated canonical LUT arrays in numerical-kernel order."""
    validate_reflectance_lut(dataset)
    reflectance = dataset["reflectance"]
    return {
        # The C++ batch kernel only uses this array's length; canonical band
        # identifiers may be strings, so pass stable one-based positions.
        "bands": np.arange(1, reflectance.sizes["band"] + 1, dtype=np.float64),
        "solar_angles": np.asarray(reflectance["solar_angle"].values),
        "lap_concentrations": np.asarray(
            reflectance["lap_concentration"].values
        ),
        "sqrt_grain_radii": np.asarray(
            reflectance["sqrt_grain_radius"].values
        ),
        "reflectances": np.asarray(reflectance.values),
    }


def _lap_concentration_to_ppm(values, units):
    normalized = _normalize_units(units)
    values = np.asarray(values)
    if normalized in {"ppm", "partspermillion", "parts_per_million"}:
        return values
    if normalized in {"fraction", "1", "dimensionless"}:
        return values * 1_000_000.0
    raise ContractError(
        "unsupported legacy MATLAB LAP concentration units "
        f"{units!r}; expected 'ppm' or 'fraction'"
    )


def _require_micrometres(values, units):
    if _normalize_units(units) not in {
        "um",
        "micron",
        "microns",
        "micrometer",
        "micrometers",
        "micrometre",
        "micrometres",
        "µm",
        "μm",
    }:
        raise ContractError(
            "unsupported legacy MATLAB grain-radius units "
            f"{units!r}; expected micrometres"
        )
    return np.asarray(values)


def _require_degrees(units):
    if _normalize_units(units) not in {"degree", "degrees", "deg"}:
        raise ContractError(
            "unsupported legacy MATLAB solar-angle units "
            f"{units!r}; expected degrees"
        )


def _normalize_units(units):
    if not isinstance(units, str) or not units.strip():
        raise ContractError("legacy MATLAB axis units must be declared explicitly")
    return "".join(units.strip().lower().split())
