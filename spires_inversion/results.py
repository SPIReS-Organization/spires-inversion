"""Construction of canonical SPIReS inversion results."""

import numpy as np
import xarray as xr

from spires_contract import ContractError
from spires_contract import conventions as contract_conventions


def build_results(
    property_values,
    spatial_reference=None,
    *,
    lap_type,
    grain_output="sqrt_grain_radius",
):
    """Build canonical results from kernel property-vector output.

    Kernel columns are snow fraction, shade fraction, LAP concentration, and
    a grain coordinate. ``grain_output="sqrt_grain_radius"`` squares the final
    column once; the legacy Dask kernel passes ``grain_output="grain_radius"``.
    NumPy and xarray/Dask inputs share the same names, dtypes, and metadata.
    """
    if grain_output not in {"sqrt_grain_radius", "grain_radius"}:
        raise ValueError(
            "grain_output must be 'sqrt_grain_radius' or 'grain_radius'"
        )

    if isinstance(property_values, xr.DataArray):
        return _build_xarray_results(
            property_values,
            lap_type=lap_type,
            grain_output=grain_output,
        )
    if spatial_reference is None:
        raise TypeError("spatial_reference is required for NumPy property values")

    values = np.asarray(property_values)
    expected_shape = spatial_reference.shape + (4,)
    if values.shape != expected_shape:
        raise ValueError(
            f"property_values shape must be {expected_shape}, got {values.shape}"
        )

    grain_values = values[..., 3]
    if grain_output == "sqrt_grain_radius":
        grain_values = np.square(grain_values)
    output_values = (
        values[..., 0],
        values[..., 1],
        values[..., 2],
        grain_values,
    )
    coordinates = {
        name: coordinate
        for name, coordinate in spatial_reference.coords.items()
        if set(coordinate.dims).issubset(contract_conventions.SPATIAL_DIMS)
    }
    data_vars = {}
    for name, array in zip(contract_conventions.RESULT_VARIABLES, output_values):
        data_vars[name] = xr.DataArray(
            np.asarray(array, dtype=np.float32),
            dims=contract_conventions.RESULT_DIMS,
            coords=coordinates,
            attrs=_variable_attrs(name, lap_type),
        )
    return xr.Dataset(data_vars)


def validate_result_structure(results):
    """Validate result schema and metadata without materializing array values.

    This lazy-safe boundary accepts canonical single scenes ``(y, x)`` and Dask
    time stacks ``(time, y, x)``. Use ``spires_contract.validate_results`` for
    full numerical validation of an eager single scene.
    """
    violations = []
    if not isinstance(results, xr.Dataset):
        raise ContractError(
            "result structure violated: results must be an xarray.Dataset"
        )

    expected_variables = set(contract_conventions.RESULT_VARIABLES)
    missing = expected_variables.difference(results.data_vars)
    if missing:
        violations.append(f"missing result variable(s): {sorted(missing)}")

    reference_dims = None
    accepted_dtypes = {
        np.dtype(dtype) for dtype in contract_conventions.ACCEPTED_DTYPES
    }
    for name in contract_conventions.RESULT_VARIABLES:
        if name not in results:
            continue
        array = results[name]
        dims = tuple(array.dims)
        if reference_dims is None:
            reference_dims = dims
        elif dims != reference_dims:
            violations.append(
                f"{name} dims {dims!r} do not match {reference_dims!r}"
            )
        if dims not in {
            contract_conventions.RESULT_DIMS,
            ("time",) + contract_conventions.RESULT_DIMS,
        }:
            violations.append(
                f"{name} dims are {dims!r}; expected "
                "('y', 'x') or ('time', 'y', 'x')"
            )
        if np.dtype(array.dtype) not in accepted_dtypes:
            violations.append(
                f"{name} dtype is {array.dtype}, expected float32"
            )
        for coordinate in contract_conventions.SPATIAL_DIMS:
            if coordinate not in array.coords:
                violations.append(f"{name} is missing coordinate {coordinate!r}")
        expected_attrs = _variable_attrs(
            name,
            contract_conventions.SUPPORTED_LAP_TYPES[0],
        )
        for attr_name, expected in expected_attrs.items():
            if array.attrs.get(attr_name) != expected:
                violations.append(
                    f"{name} attribute {attr_name!r} must be {expected!r}"
                )

    if violations:
        bullets = "\n".join(f"  - {violation}" for violation in violations)
        raise ContractError(f"result structure violated:\n{bullets}")


def _build_xarray_results(property_values, *, lap_type, grain_output):
    if "property" not in property_values.dims:
        raise ValueError("xarray property values must include a 'property' dimension")
    if property_values.sizes["property"] != 4:
        raise ValueError("xarray 'property' dimension must have length 4")
    if property_values.dims[-1] != "property":
        raise ValueError("xarray 'property' dimension must be last")

    output_values = [
        property_values.isel(property=index, drop=True)
        for index in range(4)
    ]
    if grain_output == "sqrt_grain_radius":
        output_values[3] = np.square(output_values[3])

    data_vars = {}
    for name, array in zip(contract_conventions.RESULT_VARIABLES, output_values):
        array = array.astype(np.float32)
        array.name = name
        array.attrs = _variable_attrs(name, lap_type)
        data_vars[name] = array
    return xr.Dataset(data_vars)


def _variable_attrs(name, lap_type):
    attrs = {
        "long_name": contract_conventions.RESULT_LONG_NAMES[name],
        "units": contract_conventions.RESULT_UNITS[name],
    }
    if name == "lap_concentration":
        attrs["lap_type"] = lap_type
    return attrs
