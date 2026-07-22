"""Construction of canonical SPIReS inversion results."""

import numpy as np
import xarray as xr

from spires_contract import conventions as contract_conventions


def build_results(property_values, spatial_reference, *, lap_type):
    """Build canonical results from kernel property-vector output.

    Kernel columns are snow fraction, shade fraction, LAP concentration, and
    the optimized ``sqrt_grain_radius`` LUT coordinate. Only contract-defined
    variable names are emitted.
    """
    values = np.asarray(property_values)
    expected_shape = spatial_reference.shape + (4,)
    if values.shape != expected_shape:
        raise ValueError(
            f"property_values shape must be {expected_shape}, got {values.shape}"
        )

    output_values = (
        values[..., 0],
        values[..., 1],
        values[..., 2],
        np.square(values[..., 3]),
    )
    coordinates = {
        dim: spatial_reference.coords[dim] for dim in contract_conventions.SPATIAL_DIMS
    }
    data_vars = {}
    for name, array in zip(contract_conventions.RESULT_VARIABLES, output_values):
        attrs = {
            "long_name": contract_conventions.RESULT_LONG_NAMES[name],
            "units": contract_conventions.RESULT_UNITS[name],
        }
        if name == "lap_concentration":
            attrs["lap_type"] = lap_type
        data_vars[name] = xr.DataArray(
            np.asarray(array, dtype=np.float32),
            dims=contract_conventions.RESULT_DIMS,
            coords=coordinates,
            attrs=attrs,
        )
    return xr.Dataset(data_vars)
