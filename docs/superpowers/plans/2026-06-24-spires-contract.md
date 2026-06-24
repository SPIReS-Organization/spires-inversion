# spires-contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone `spires-contract` package: an xarray-based validator library that defines the data interfaces between SPIReS packages, with the I/O→inversion spectra contract fully implemented and `lut`/`r0`/`results` stubbed.

**Architecture:** A small `src/`-layout Python package depending only on numpy + xarray. A shared `conventions` module holds canonical dim names, dtype rules, and units vocabulary. A `_validate` module provides collect-all-violations machinery and a `ContractError`. Boundary modules (`spectra` first) declare required dims/coords/dtype and expose `validate_*` (raises, listing every violation) and `conform_*` (normalizer that transposes/casts to canonical form) functions.

**Tech Stack:** Python ≥3.9, numpy, xarray, pytest. Conda env: `spipy14`. New repo lives at `/home/griessban/spires-contract` (sibling of `SpiPy`), pushed to `github.com/SPIReS-Organization/spires-contract`.

**Attribution:** Commits authored by `niklas <niklas.griessbaum@leidos.com>`. Do NOT add a `Co-Authored-By: Claude` trailer on any commit in this plan (infrastructure seed commits — see design doc).

**Conventions reference (from existing `SpiPy/spires/invert.py`):**
- Spectra dims: `(y, x, band)`; solar angles: `(y, x)`; results: `(y, x, 4)`.
- LUT dims: `(band, solar_angle, dust_concentration, grain_size)`.
- The C++ inversion layer requires `float64` (`np.double`) arrays.
- Result vector order: `[fsca, fshade, dust_concentration, grain_size]` (units: unitless, unitless, ppm, μm).

---

## File Structure

```
spires-contract/                       # repo root (/home/griessban/spires-contract)
  pyproject.toml                        # package metadata, deps: numpy, xarray; dev: pytest
  README.md                             # what the package is, install, usage
  .gitignore                            # python ignores
  src/spires_contract/
    __init__.py                         # version + public re-exports
    conventions.py                      # canonical dim names, dtype, units vocabulary
    _validate.py                        # ContractError + violation-collecting helpers
    spectra.py                          # I/O -> inversion contract (FULL)
    lut.py                              # LUT -> inversion contract (STUB)
    r0.py                               # r0 -> inversion contract (STUB)
    results.py                          # inversion -> postprocess contract (STUB)
  tests/
    test_conventions.py
    test_validate.py
    test_spectra.py
```

**Responsibilities:**
- `conventions.py` — single source of truth for dimension names and dtype. No logic beyond constants/tiny helpers. Imported by every boundary module.
- `_validate.py` — reusable validation primitives (check dims present, check dtype, check coords) that accumulate messages, plus `ContractError`. No knowledge of any specific boundary.
- `spectra.py` — the spectra boundary: composes `_validate` primitives into `validate_target_spectra` / `validate_background_spectra` / `validate_solar_angles`, and `conform_*` normalizers.
- `lut.py`, `r0.py`, `results.py` — stubs that raise `NotImplementedError` with a docstring sketch of the intended spec.

---

## Task 1: Scaffold the repo and packaging

**Files:**
- Create: `/home/griessban/spires-contract/pyproject.toml`
- Create: `/home/griessban/spires-contract/.gitignore`
- Create: `/home/griessban/spires-contract/README.md`
- Create: `/home/griessban/spires-contract/src/spires_contract/__init__.py`

- [ ] **Step 1: Create the repo directory and init git**

Run:
```bash
mkdir -p /home/griessban/spires-contract/src/spires_contract /home/griessban/spires-contract/tests
cd /home/griessban/spires-contract && git init -b master
```
Expected: `Initialized empty Git repository in /home/griessban/spires-contract/.git/`

- [ ] **Step 2: Write `pyproject.toml`**

Create `/home/griessban/spires-contract/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "spires-contract"
version = "0.1.0"
description = "Data-interface contracts (xarray schemas + validators) for the SPIReS package family"
readme = { file = "README.md", content-type = "text/markdown" }
requires-python = ">=3.9"
license = { text = "MIT" }
authors = [{ name = "Niklas Griessbaum" }]
keywords = ["snow", "remote sensing", "xarray", "data contract", "validation"]
dependencies = [
    "numpy",
    "xarray",
]

[project.optional-dependencies]
test = ["pytest"]

[project.urls]
Homepage = "https://github.com/SPIReS-Organization/spires-contract"
Repository = "https://github.com/SPIReS-Organization/spires-contract"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 3: Write `.gitignore`**

Create `/home/griessban/spires-contract/.gitignore`:
```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.pytest_cache/
.tox/
.venv/
```

- [ ] **Step 4: Write a minimal `README.md`**

Create `/home/griessban/spires-contract/README.md`:
```markdown
# spires-contract

Data-interface contracts for the [SPIReS](https://github.com/SPIReS-Organization)
package family. Defines, as executable xarray validators, the array shapes,
dimension names, dtypes, and coordinate conventions that flow between SPIReS
packages (I/O, R_0 production, inversion, postprocessing).

Depends only on `numpy` and `xarray`.

## Install

```bash
pip install spires-contract
```

## Usage

```python
import spires_contract.spectra as spectra

spectra.validate_target_spectra(da)   # raises ContractError listing all violations
da = spectra.conform_target_spectra(da)  # transpose/cast to canonical (y, x, band) float64
```

## Boundaries

| Module                 | Boundary                  | Status      |
|------------------------|---------------------------|-------------|
| `spires_contract.spectra` | I/O → inversion        | implemented |
| `spires_contract.r0`      | R_0 → inversion        | stub        |
| `spires_contract.results` | inversion → postprocess | stub        |
```

- [ ] **Step 5: Write `__init__.py`**

Create `/home/griessban/spires-contract/src/spires_contract/__init__.py`:
```python
"""spires-contract: data-interface contracts for the SPIReS package family."""

__version__ = "0.1.0"

from spires_contract._validate import ContractError

__all__ = ["ContractError", "__version__"]
```

- [ ] **Step 6: Verify the package imports**

Run:
```bash
cd /home/griessban/spires-contract && conda run -n spipy14 pip install -e . && conda run -n spipy14 python -c "import spires_contract; print(spires_contract.__version__)"
```
Expected: prints `0.1.0` (after a successful editable install).

Note: Step 6 imports `ContractError` from `_validate`, which is created in Task 3. To keep this task self-contained, temporarily comment out the `from spires_contract._validate import ContractError` line and `ContractError` in `__all__` for this verification, then restore it in Task 3 Step 5. (Alternatively, run Step 6's verification at the end of Task 3.)

- [ ] **Step 7: Commit**

```bash
cd /home/griessban/spires-contract
git add pyproject.toml .gitignore README.md src/spires_contract/__init__.py
git commit -m "Scaffold spires-contract package"
```

---

## Task 2: Conventions module

**Files:**
- Create: `/home/griessban/spires-contract/src/spires_contract/conventions.py`
- Test: `/home/griessban/spires-contract/tests/test_conventions.py`

- [ ] **Step 1: Write the failing test**

Create `/home/griessban/spires-contract/tests/test_conventions.py`:
```python
import numpy as np
from spires_contract import conventions as c


def test_canonical_spectra_dims():
    assert c.SPECTRA_DIMS == ("y", "x", "band")


def test_canonical_solar_angle_dims():
    assert c.SOLAR_ANGLE_DIMS == ("y", "x")


def test_canonical_lut_dims():
    assert c.LUT_DIMS == ("band", "solar_angle", "dust_concentration", "grain_size")


def test_result_variables_order():
    assert c.RESULT_VARIABLES == ("fsca", "fshade", "dust_concentration", "grain_size")


def test_required_dtype_is_float64():
    assert c.REQUIRED_DTYPE == np.float64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 pytest tests/test_conventions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spires_contract.conventions'` (or AttributeError).

- [ ] **Step 3: Write minimal implementation**

Create `/home/griessban/spires-contract/src/spires_contract/conventions.py`:
```python
"""Canonical naming and dtype conventions shared across all SPIReS contracts.

Single source of truth so every boundary module (spectra, r0, results) speaks
the same dimension-name and dtype vocabulary.
"""

import numpy as np

# Spatial + spectral spectra arrays (target / background reflectance).
SPECTRA_DIMS = ("y", "x", "band")

# Per-pixel solar zenith angle (degrees).
SOLAR_ANGLE_DIMS = ("y", "x")

# Reflectance lookup table produced from Mie theory.
LUT_DIMS = ("band", "solar_angle", "dust_concentration", "grain_size")

# Inversion output vector, in this order, along the trailing result axis.
RESULT_VARIABLES = ("fsca", "fshade", "dust_concentration", "grain_size")

# The C++/SWIG inversion layer requires double-precision arrays.
REQUIRED_DTYPE = np.float64
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 pytest tests/test_conventions.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/griessban/spires-contract
git add src/spires_contract/conventions.py tests/test_conventions.py
git commit -m "Add shared naming/dtype conventions"
```

---

## Task 3: Validation machinery (`_validate`)

**Files:**
- Create: `/home/griessban/spires-contract/src/spires_contract/_validate.py`
- Modify: `/home/griessban/spires-contract/src/spires_contract/__init__.py`
- Test: `/home/griessban/spires-contract/tests/test_validate.py`

- [ ] **Step 1: Write the failing test**

Create `/home/griessban/spires-contract/tests/test_validate.py`:
```python
import numpy as np
import pytest
import xarray as xr

from spires_contract._validate import (
    ContractError,
    check_dims_present,
    check_dtype,
    check_coords_present,
    raise_if_violations,
)


def _da(dims, coords=None, dtype=np.float64):
    shape = tuple(2 for _ in dims)
    return xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims, coords=coords or {})


def test_check_dims_present_no_violation():
    da = _da(("y", "x", "band"))
    assert check_dims_present(da, ("y", "x", "band")) == []


def test_check_dims_present_reports_missing():
    da = _da(("y", "x"))
    msgs = check_dims_present(da, ("y", "x", "band"))
    assert len(msgs) == 1
    assert "band" in msgs[0]


def test_check_dtype_no_violation():
    da = _da(("y", "x"), dtype=np.float64)
    assert check_dtype(da, np.float64) == []


def test_check_dtype_reports_mismatch():
    da = _da(("y", "x"), dtype=np.float32)
    msgs = check_dtype(da, np.float64)
    assert len(msgs) == 1
    assert "float64" in msgs[0] and "float32" in msgs[0]


def test_check_coords_present_reports_missing():
    da = _da(("y", "x", "band"))  # no coords assigned
    msgs = check_coords_present(da, ("band",))
    assert len(msgs) == 1
    assert "band" in msgs[0]


def test_check_coords_present_no_violation():
    da = _da(("band",), coords={"band": [1, 2]})
    assert check_coords_present(da, ("band",)) == []


def test_raise_if_violations_raises_with_all_messages():
    with pytest.raises(ContractError) as exc:
        raise_if_violations("target_spectra", ["missing dim 'band'", "dtype is float32"])
    text = str(exc.value)
    assert "target_spectra" in text
    assert "missing dim 'band'" in text
    assert "dtype is float32" in text


def test_raise_if_violations_silent_when_empty():
    raise_if_violations("target_spectra", [])  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 pytest tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spires_contract._validate'`.

- [ ] **Step 3: Write minimal implementation**

Create `/home/griessban/spires-contract/src/spires_contract/_validate.py`:
```python
"""Reusable validation primitives shared by all boundary contracts.

Each `check_*` function returns a list of human-readable violation strings
(empty list = conforms). Boundary modules compose these and call
`raise_if_violations` so a producer gets ONE error listing EVERY problem.
"""

import numpy as np


class ContractError(ValueError):
    """Raised when a data array violates a SPIReS boundary contract."""


def check_dims_present(da, required_dims):
    """Return a violation per required dim missing from `da.dims`."""
    return [
        f"missing required dimension {dim!r} (found dims: {tuple(da.dims)})"
        for dim in required_dims
        if dim not in da.dims
    ]


def check_dtype(da, required_dtype):
    """Return a violation if `da` is not the required dtype."""
    if np.dtype(da.dtype) != np.dtype(required_dtype):
        return [
            f"dtype is {np.dtype(da.dtype)}, expected {np.dtype(required_dtype)}"
        ]
    return []


def check_coords_present(da, required_coords):
    """Return a violation per required coordinate missing from `da.coords`."""
    return [
        f"missing required coordinate {coord!r} (found coords: {tuple(da.coords)})"
        for coord in required_coords
        if coord not in da.coords
    ]


def raise_if_violations(contract_name, violations):
    """Raise a single ContractError listing all violations, if any."""
    if violations:
        bullets = "\n".join(f"  - {v}" for v in violations)
        raise ContractError(
            f"{contract_name} contract violated:\n{bullets}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 pytest tests/test_validate.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Ensure `__init__.py` re-exports `ContractError`**

Confirm `/home/griessban/spires-contract/src/spires_contract/__init__.py` contains (restore if commented out in Task 1 Step 6):
```python
"""spires-contract: data-interface contracts for the SPIReS package family."""

__version__ = "0.1.0"

from spires_contract._validate import ContractError

__all__ = ["ContractError", "__version__"]
```

- [ ] **Step 6: Verify top-level import works**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 python -c "from spires_contract import ContractError; print(ContractError)"`
Expected: prints `<class 'spires_contract._validate.ContractError'>`.

- [ ] **Step 7: Commit**

```bash
cd /home/griessban/spires-contract
git add src/spires_contract/_validate.py src/spires_contract/__init__.py tests/test_validate.py
git commit -m "Add validation primitives and ContractError"
```

---

## Task 4: Spectra contract — validators

**Files:**
- Create: `/home/griessban/spires-contract/src/spires_contract/spectra.py`
- Test: `/home/griessban/spires-contract/tests/test_spectra.py`

- [ ] **Step 1: Write the failing test**

Create `/home/griessban/spires-contract/tests/test_spectra.py`:
```python
import numpy as np
import pytest
import xarray as xr

from spires_contract import spectra
from spires_contract._validate import ContractError


def make_target(dims=("y", "x", "band"), dtype=np.float64, with_band_coord=True):
    shape = tuple({"y": 3, "x": 4, "band": 5}[d] for d in dims)
    coords = {"band": np.arange(shape[dims.index("band")])} if with_band_coord and "band" in dims else {}
    return xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims, coords=coords)


def make_solar(dims=("y", "x"), dtype=np.float64):
    shape = tuple({"y": 3, "x": 4}[d] for d in dims)
    return xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims)


def test_validate_target_spectra_accepts_valid():
    spectra.validate_target_spectra(make_target())  # must not raise


def test_validate_target_spectra_rejects_missing_band_dim():
    da = make_target(dims=("y", "x"), with_band_coord=False)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "band" in str(exc.value)


def test_validate_target_spectra_rejects_wrong_dtype():
    da = make_target(dtype=np.float32)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "float64" in str(exc.value)


def test_validate_target_spectra_rejects_missing_band_coord():
    da = make_target(with_band_coord=False)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "band" in str(exc.value)


def test_validate_target_spectra_collects_multiple_violations():
    # wrong dtype AND missing band coordinate -> both reported in one error
    da = make_target(dtype=np.float32, with_band_coord=False)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    text = str(exc.value)
    assert "float64" in text
    assert "band" in text


def test_validate_target_spectra_accepts_any_dim_order():
    # transposed order is still valid (validator checks presence, not order)
    da = make_target(dims=("band", "y", "x"))
    spectra.validate_target_spectra(da)  # must not raise


def test_validate_background_spectra_accepts_valid():
    spectra.validate_background_spectra(make_target())  # same shape rules as target


def test_validate_solar_angles_accepts_valid():
    spectra.validate_solar_angles(make_solar())


def test_validate_solar_angles_rejects_band_dim():
    da = xr.DataArray(np.zeros((3, 4, 5), dtype=np.float64), dims=("y", "x", "band"))
    with pytest.raises(ContractError):
        spectra.validate_solar_angles(da)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 pytest tests/test_spectra.py -v`
Expected: FAIL with `ImportError: cannot import name 'spectra'` (or ModuleNotFound).

- [ ] **Step 3: Write minimal implementation**

Create `/home/griessban/spires-contract/src/spires_contract/spectra.py`:
```python
"""I/O -> inversion boundary contract: target/background spectra + solar angles.

Canonical forms (see conventions):
- target/background spectra: dims (y, x, band), float64, with a `band` coordinate
- solar angles:              dims (y, x),       float64

`validate_*` raises ContractError listing every violation. `conform_*`
(see below) transposes/casts a nearly-conforming array into canonical form.
"""

from spires_contract import conventions as c
from spires_contract._validate import (
    check_coords_present,
    check_dims_present,
    check_dtype,
    raise_if_violations,
)


def _validate_spectra(da, contract_name):
    violations = []
    violations += check_dims_present(da, c.SPECTRA_DIMS)
    violations += check_dtype(da, c.REQUIRED_DTYPE)
    violations += check_coords_present(da, ("band",))
    raise_if_violations(contract_name, violations)


def validate_target_spectra(da):
    """Validate mixed target reflectance spectra. Raises ContractError."""
    _validate_spectra(da, "target_spectra")


def validate_background_spectra(da):
    """Validate background (R_0) reflectance spectra. Raises ContractError."""
    _validate_spectra(da, "background_spectra")


def validate_solar_angles(da):
    """Validate per-pixel solar zenith angles. Raises ContractError."""
    violations = []
    violations += check_dims_present(da, c.SOLAR_ANGLE_DIMS)
    violations += check_dtype(da, c.REQUIRED_DTYPE)
    # solar angles are 2-D: a band dimension is a violation
    if "band" in da.dims:
        violations.append(
            f"unexpected dimension 'band' for solar angles (dims: {tuple(da.dims)})"
        )
    raise_if_violations("solar_angles", violations)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 pytest tests/test_spectra.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/griessban/spires-contract
git add src/spires_contract/spectra.py tests/test_spectra.py
git commit -m "Add spectra boundary validators"
```

---

## Task 5: Spectra contract — conform/normalize helpers

**Files:**
- Modify: `/home/griessban/spires-contract/src/spires_contract/spectra.py`
- Test: `/home/griessban/spires-contract/tests/test_spectra.py` (append)

- [ ] **Step 1: Write the failing test (append to test_spectra.py)**

Append to `/home/griessban/spires-contract/tests/test_spectra.py`:
```python
def test_conform_target_spectra_transposes_to_canonical_order():
    da = make_target(dims=("band", "y", "x"))
    out = spectra.conform_target_spectra(da)
    assert out.dims == ("y", "x", "band")


def test_conform_target_spectra_casts_dtype():
    da = make_target(dtype=np.float32)
    out = spectra.conform_target_spectra(da)
    assert out.dtype == np.float64


def test_conform_target_spectra_output_passes_validation():
    da = make_target(dims=("band", "y", "x"), dtype=np.float32)
    out = spectra.conform_target_spectra(da)
    spectra.validate_target_spectra(out)  # must not raise


def test_conform_solar_angles_transposes_and_casts():
    da = xr.DataArray(np.zeros((4, 3), dtype=np.float32), dims=("x", "y"))
    out = spectra.conform_solar_angles(da)
    assert out.dims == ("y", "x")
    assert out.dtype == np.float64


def test_conform_target_spectra_raises_when_dim_absent():
    # conform cannot fix a genuinely missing dimension; it should surface that
    da = make_target(dims=("y", "x"), with_band_coord=False)
    with pytest.raises(ContractError):
        spectra.conform_target_spectra(da)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 pytest tests/test_spectra.py -k conform -v`
Expected: FAIL with `AttributeError: module 'spires_contract.spectra' has no attribute 'conform_target_spectra'`.

- [ ] **Step 3: Write minimal implementation (append to spectra.py)**

Append to `/home/griessban/spires-contract/src/spires_contract/spectra.py`:
```python
def _conform_spectra(da, contract_name):
    # A missing dimension cannot be repaired by transpose/cast — fail clearly.
    missing = check_dims_present(da, c.SPECTRA_DIMS)
    raise_if_violations(contract_name, missing)
    return da.transpose(*c.SPECTRA_DIMS).astype(c.REQUIRED_DTYPE)


def conform_target_spectra(da):
    """Return target spectra transposed to (y, x, band) and cast to float64."""
    return _conform_spectra(da, "target_spectra")


def conform_background_spectra(da):
    """Return background spectra transposed to (y, x, band) and cast to float64."""
    return _conform_spectra(da, "background_spectra")


def conform_solar_angles(da):
    """Return solar angles transposed to (y, x) and cast to float64."""
    missing = check_dims_present(da, c.SOLAR_ANGLE_DIMS)
    raise_if_violations("solar_angles", missing)
    return da.transpose(*c.SOLAR_ANGLE_DIMS).astype(c.REQUIRED_DTYPE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 pytest tests/test_spectra.py -v`
Expected: PASS (14 passed — 9 from Task 4 + 5 new).

- [ ] **Step 5: Commit**

```bash
cd /home/griessban/spires-contract
git add src/spires_contract/spectra.py tests/test_spectra.py
git commit -m "Add spectra conform/normalize helpers"
```

---

## Task 6: Stub the lut, r0, and results boundaries

**Files:**
- Create: `/home/griessban/spires-contract/src/spires_contract/lut.py`
- Create: `/home/griessban/spires-contract/src/spires_contract/r0.py`
- Create: `/home/griessban/spires-contract/src/spires_contract/results.py`

- [ ] **Step 1: Write `lut.py` stub**

Create `/home/griessban/spires-contract/src/spires_contract/lut.py`:
```python
"""LUT -> inversion boundary contract (STUB — not yet implemented).

Planned spec: a Mie-theory reflectance lookup table with dims
(band, solar_angle, dust_concentration, grain_size) — see conventions.LUT_DIMS —
float64, with a coordinate present for each of the four dimensions (the
interpolator reads coordinate values to locate query points). To be implemented
when the spires-lut package is built.
"""

from spires_contract._validate import ContractError  # noqa: F401  (re-exported for future use)


def validate_lut(da):
    """Validate a reflectance lookup table DataArray. Not yet implemented."""
    raise NotImplementedError(
        "The lut -> inversion contract is not implemented yet; "
        "it will be added when the spires-lut package is built."
    )
```

- [ ] **Step 2: Write `r0.py` stub**

Create `/home/griessban/spires-contract/src/spires_contract/r0.py`:
```python
"""R_0 -> inversion boundary contract (STUB — not yet implemented).

Planned spec: background (snow-free) reflectance R_0 with dims (y, x, band),
float64, matching the spatial grid and band coordinate of the target spectra it
will be paired with. To be implemented when the spires-r0 package is built.
"""

from spires_contract._validate import ContractError  # noqa: F401  (re-exported for future use)


def validate_r0(da):
    """Validate an R_0 background reflectance array. Not yet implemented."""
    raise NotImplementedError(
        "The r0 -> inversion contract is not implemented yet; "
        "it will be added when the spires-r0 package is built."
    )
```

- [ ] **Step 3: Write `results.py` stub**

Create `/home/griessban/spires-contract/src/spires_contract/results.py`:
```python
"""inversion -> postprocess boundary contract (STUB — not yet implemented).

Planned spec: inversion output with dims (y, x) per variable, one variable each
for fsca, fshade, dust_concentration (ppm), grain_size (μm) — see
conventions.RESULT_VARIABLES — likely as an xarray.Dataset. To be implemented
when the spires-postprocess package is built.
"""

from spires_contract._validate import ContractError  # noqa: F401  (re-exported for future use)


def validate_results(ds):
    """Validate an inversion results Dataset. Not yet implemented."""
    raise NotImplementedError(
        "The inversion -> postprocess contract is not implemented yet; "
        "it will be added when the spires-postprocess package is built."
    )
```

- [ ] **Step 4: Verify the stubs import and raise as expected**

Run:
```bash
cd /home/griessban/spires-contract && conda run -n spipy14 python -c "
import spires_contract.lut as lut, spires_contract.r0 as r0, spires_contract.results as res
for fn in [lut.validate_lut, r0.validate_r0, res.validate_results]:
    try:
        fn(None)
    except NotImplementedError as e:
        print('OK:', str(e)[:40])
"
```
Expected: three `OK:` lines.

- [ ] **Step 5: Commit**

```bash
cd /home/griessban/spires-contract
git add src/spires_contract/lut.py src/spires_contract/r0.py src/spires_contract/results.py
git commit -m "Stub lut, r0, and results boundary contracts"
```

---

## Task 7: Full test run and push to the org

**Files:** none (verification + remote setup)

- [ ] **Step 1: Run the full test suite**

Run: `cd /home/griessban/spires-contract && conda run -n spipy14 pytest -v`
Expected: PASS (27 tests: 5 conventions + 8 validate + 14 spectra). 0 failures.

- [ ] **Step 2: Verify a clean editable install + import surface**

Run:
```bash
cd /home/griessban/spires-contract && conda run -n spipy14 pip install -e . >/dev/null && conda run -n spipy14 python -c "
import spires_contract as sc
import spires_contract.spectra as spectra
print('version:', sc.__version__)
print('ContractError:', sc.ContractError.__name__)
print('spectra fns:', [f for f in dir(spectra) if f.startswith(('validate_', 'conform_'))])
"
```
Expected: prints version `0.1.0`, `ContractError`, and the six `validate_*`/`conform_*` function names.

- [ ] **Step 3: Create the GitHub repo under the org**

Run (requires `gh` authenticated; `spipy14` env):
```bash
cd /home/griessban/spires-contract && conda run -n spipy14 gh repo create SPIReS-Organization/spires-contract --private --source=. --remote=org --description "Data-interface contracts (xarray schemas + validators) for the SPIReS package family"
```
Expected: repo created; remote `org` added. If `gh` is unavailable or unauthenticated, STOP and ask the user to create the empty repo, then `git remote add org https://github.com/SPIReS-Organization/spires-contract.git`.

Note: `--private` vs `--public` — confirm with the user before running (default to `--private` if unsure).

- [ ] **Step 4: Push**

Run: `cd /home/griessban/spires-contract && git push -u org master`
Expected: branch `master` pushed; upstream set.

- [ ] **Step 5: Confirm the remote has the commits**

Run: `cd /home/griessban/spires-contract && git log --oneline origin/master 2>/dev/null || git log --oneline org/master`
Expected: lists all task commits, authored by `niklas` with no `Co-Authored-By: Claude` trailer.

---

## Self-Review Notes

- **Spec coverage:** `conventions` (Task 2), `_validate`+`ContractError` (Task 3), full spectra boundary with collect-all-violations validators (Task 4) and conform/transpose-absorbing-`speedy_invert_xarray`-logic (Task 5), `lut`/`r0`/`results` stubs (Task 6), src-layout + numpy/xarray-only deps + no-Claude-attribution (Tasks 1, 7). All map to the design doc's Section B.
- **Type/name consistency:** `ContractError`, `check_dims_present`/`check_dtype`/`check_coords_present`/`raise_if_violations`, and `validate_*`/`conform_*` names are identical across tasks and tests.
- **Convention values** (`SPECTRA_DIMS`, `REQUIRED_DTYPE=float64`, etc.) verified against `SpiPy/spires/invert.py` (`speedy_invert_xarray`: dims `(y,x,band)`/`(y,x)`, `np.double` results, LUT dims `(band, solar_angle, dust_concentration, grain_size)`).
- **Open item surfaced for executor:** Task 7 Step 3 pauses for public/private choice and `gh` availability rather than guessing.
```
