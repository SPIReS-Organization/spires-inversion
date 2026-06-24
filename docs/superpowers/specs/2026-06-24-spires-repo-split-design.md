# SPIReS Repository Split — Design

**Date:** 2026-06-24
**Status:** Approved topology; first deliverable = `spires-contract`

## Goal

Split the current monolithic `SpiPy` package into a family of focused,
independently installable packages under the GitHub organization
[`SPIReS-Organization`](https://github.com/SPIReS-Organization). Each package
has one clear purpose, its own CI, and communicates with the others only
through **contract-validated xarray data** — never by importing another
package's internals.

This document captures (A) the org-wide topology and migration plan we are
committing to, and (B) the design of the **first repo we build**,
`spires-contract`. Each subsequent repo gets its own spec → plan → implement
cycle.

## A. Org Topology

Seven repos, each a pip-installable package. Dependencies point downward only
(no cycles):

```
                       spires-contract        ← numpy + xarray only
                       (spectra, lut, r0,
                        results, conventions)
                  /      /     |      \      \
                 /      /      |       \      \
       spires-io  spires-lut spires-r0  spires-inversion   ← each pins a contract version
      (rioxarray, (xarray,   (xarray,   (numpy, scipy,
       pyproj,     h5py)      rioxarray) C++/SWIG/nlopt)
       gdal)          \        |         /
                       \       |        /
                      spires-postprocess          ← clouds (scipy) + trees (torch, lama)
                      
                            |
                     (consumes inversion
                      results contract)
```

| Repo                 | Role                                                         | Heavy deps              |
|----------------------|--------------------------------------------------------------|-------------------------|
| `spires-contract`    | All boundary schemas + shared conventions/validators         | numpy, xarray           |
| `spires-inversion`   | Core: `invert`, `interpolator`, C++/SWIG, `utol`, dask       | numpy, scipy, nlopt     |
| `spires-io`          | MODIS/Sentinel-2/Landsat loaders, reproject, transforms      | rioxarray, pyproj, gdal |
| `spires-lut`         | Create / read / write reflectance lookup tables (LUTs)       | xarray, h5py            |
| `spires-r0`          | Background (R_0) reflectance production                      | xarray, rioxarray       |
| `spires-postprocess` | Cloud gap-fill, tree masking/inpainting                      | scipy, torch, lama      |

### Principles

- The contract is the **only** shared dependency. No package imports another
  package's internals; they exchange contract-validated xarray objects.
- Each boundary (io→inversion, lut→inversion, r0→inversion,
  inversion→postprocess) is a **submodule** inside the single `spires-contract`
  package — not a separate repo. This keeps shared conventions (dim names,
  ordering, units, dtype rules) consistent and un-duplicated.
- Build **bottom-up**, starting with `spires-contract`.

## Migration Plan (GitHub-side)

The current repo is being promoted to the canonical `spires-inversion`,
preserving history, issues, PRs, and releases.

- [x] **Transfer** `NiklasPhabian/SpiPy` → `SPIReS-Organization` (done by user).
- [x] **Rename** transferred repo to `spires-inversion` (done by user).
- [x] **Re-point local remote**: `org` →
      `https://github.com/SPIReS-Organization/spires-inversion.git`.
      (`origin` → Leidos GitLab and `upstream` → `edwardbair/SpiPy` left intact.)

### Follow-ups (non-blocking)

- The org repo still shows **"forked from edwardbair/SpiPy."** The transfer
  carried the fork relationship along. Consequences: new PRs may default their
  base to Ned's repo; the repo may be de-emphasized in GitHub search; deleting
  the parent re-parents the fork network. To make the org repo unambiguously
  canonical, file a GitHub Support **fork-detach** request against
  `SPIReS-Organization/spires-inversion`. Not urgent; blocks nothing.
- **Coordinate with Ned Bair** on retiring/archiving `edwardbair/SpiPy` (with a
  README pointer to the org). User is raising this with Ned.
- Decide whether the **Leidos GitLab mirror** (`origin`) stays as a push mirror
  or is retired.

## B. First Deliverable — `spires-contract`

A tiny, standalone, xarray-based package that defines and validates the data
interfaces between SPIReS packages. **Dependencies: numpy + xarray only** — no
scipy, rioxarray, gdal, or torch.

### Why xarray, not numpy-only

xarray carries named dimensions, coordinates, and `attrs` (units, CRS). An
xarray-based contract can validate the things that actually matter at the
boundaries — dimension names and order, coordinate presence, units metadata —
not just raw array shape.

### Package layout

```
spires-contract/                 # repo (hyphen)
  pyproject.toml                 # deps: numpy, xarray
  src/spires_contract/           # import name (underscore), src-layout
    __init__.py
    conventions.py               # shared: canonical dim names, ordering, units, dtype rules
    _validate.py                 # core validator machinery (collect-all-violations)
    spectra.py                   # I/O → inversion: target/background spectra  ← BUILD FIRST
    lut.py                       # LUT → inversion: reflectance lookup table (STUB)
    r0.py                        # r0 → inversion: background reflectance (STUB)
    results.py                   # inversion → postprocess results (STUB)
  tests/
    test_spectra.py
    test_conventions.py
  README.md
```

### The first boundary: I/O → inversion (spectra)

Derived from the existing `speedy_invert_xarray` docstring and call in
`spires/invert.py` (the informal contract that exists today):

| Array                 | dims                                                  | units / notes                |
|-----------------------|-------------------------------------------------------|------------------------------|
| `spectra_targets`     | `(y, x, band)`                                         | reflectance, mixed spectra   |
| `spectra_backgrounds` | `(y, x, band)`                                         | reflectance, R_0             |
| `obs_solar_angles`    | `(y, x)`                                               | degrees                      |
| `lut` (model side)    | `(band, solar_angle, dust_concentration, grain_size)` | reflectance LUT              |
| `results` (output)    | `(y, x, 4)` → fsca, fshade, dust (ppm), grain (μm)     | inversion output             |

### What a contract module provides (e.g. `spectra.py`)

- A **declarative spec**: required dims `(y, x, band)`, floating dtype, required
  coordinates (e.g. `band`), optional CRS attr.
- A **validator** `validate_target_spectra(da) -> None` that raises
  `ContractError` listing **every** violation (wrong dims, missing coords, bad
  dtype) — not just the first — so a producer gets one actionable error.
- A lightweight **normalizer/constructor** helper (e.g. transpose to canonical
  order) so producers can conform easily. This absorbs the "automatically
  transposed if needed" logic currently buried in `speedy_invert_xarray`.

### Design decisions

- **Validators raise and collect all violations** (custom `ContractError`),
  giving producers one actionable message rather than whack-a-mole.
- **Conventions centralized** in `conventions.py` so `lut`/`r0`/`results` reuse
  the same dim-name/units/dtype vocabulary — the reason one repo with submodules
  beats one-repo-per-boundary. The LUT dims
  `(band, solar_angle, dust_concentration, grain_size)` already live there, so
  the `lut` boundary largely formalizes an existing convention.
- **Validation style:** plain functions raising a custom `ContractError`, with
  **no schema-library dependency** (no pydantic/pandera). Keeps the package
  numpy+xarray-only and trivial to depend on.
- **Scope for today:** fully build `conventions` + `_validate` + `spectra`
  (with tests, TDD). `lut.py`, `r0.py`, and `results.py` are stubs with their
  spec sketched, to be filled in when those repos are built.
- **TDD-friendly:** each contract is pure validation over small synthetic
  `DataArray`s — fast, no fixtures, no LUT files.

### How the contract is used from both sides

The contract is defined once and used to test **both directions** of each boundary:

- **Producer side** (e.g. `spires-io`): test that its output passes the
  validator — `validate_target_spectra(da)` must not raise.
- **Consumer side** (e.g. `spires-inversion`): two obligations. (1) It must
  accept *anything the contract permits* — its tests build conforming inputs,
  certify them with `validate_target_spectra`, and assert the consumer handles
  them (e.g. any legal dimension order). (2) As a producer of downstream data
  (`results`), it gets a producer-side test against that boundary's validator.

A contract validates **data**, not **behavior**: it eliminates shape/dtype/dim-
naming mismatches at the seams, but each package still owns its own numerical/
correctness tests.

**Decision:** the contract stays minimal — `validate_*` + `conform_*` only. It
will NOT ship shared "example builder" fixtures; each consumer/producer builds
its own test fixtures and certifies them with the validators.

### Confirmed conventions

- `src/` layout, import name `spires_contract`, repo name `spires-contract`.
- Plain-function validators + `ContractError` (no schema library).
- Build `conventions` + `_validate` + `spectra` fully; stub `lut`/`r0`/`results`.

## Attribution

Infrastructure / repo-split commits are authored by the user
(`niklas <niklas.griessbaum@leidos.com>`) with **no `Co-Authored-By: Claude`
trailer** — not to withhold credit, but to avoid sparking pointless discussion
with collaborators on the foundational seed commits. Claude attribution **is**
wanted for the normal PR-driven workflow that follows, where the trailer is
added back.

## Out of Scope (future spec → plan cycles)

- Extracting `spires-inversion` core out of the current package.
- `spires-io`, `spires-lut`, `spires-r0`, `spires-postprocess` repos.
- Additional contract submodules beyond `spectra` (i.e. `lut`, `r0`, `results`).
- Pansharpening (nb 02), cloud masking (nb 04), animations (nb 08) placement.
```
