# Test Data Files

This directory contains test data for the SpiPy package.

## Files in Repository (via Git LFS)

### Lookup Tables

- **lut_sentinel2b_b2to12_3um_dust.mat** (70 MB)
  - Lookup table for Sentinel-2B bands 2-12 with dust parameters
  - Essential for all Sentinel-2 tests
  - Also available on Zenodo (see below)

- **lut_HLSS30_b1to13_3um_dust.mat** (101 MB)
  - Lookup table for HLS (Harmonized Landsat Sentinel-2) bands 1-13 with dust parameters
  - For HLS30 product testing

- **lut_modis_b1to7_3um_dust.mat** (537 MB)
  - Lookup table for MODIS bands 1-7 with dust parameters
  - For MODIS testing

- **lut_oli_b1to7_3um_dust.mat** (55 MB)
  - Lookup table for Landsat OLI bands 1-7 with dust parameters
  - For Landsat 8/9 testing

### Test Imagery Subsets

- **sentinel_r_subset.nc** (2.85 MB)
  - Small spatial subset (50×50 pixels) of full reflectance data
  - For quick integration tests

- **sentinel_r0_subset.nc** (1.44 MB)
  - Small spatial subset (50×50 pixels) of background reflectance
  - For quick integration tests

## Large Files (Download from Zenodo)

Full-resolution test data available on Zenodo:

### Lookup Tables
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18701286.svg)](https://doi.org/10.5281/zenodo.18701286)

**Note:** All lookup tables are included in the repository via Git LFS. Download from Zenodo if you have LFS quota issues or prefer direct downloads.

- **lut_sentinel2b_b2to12_3um_dust.mat** (70 MB)
  - Sentinel-2B lookup table (also in repository via LFS)
  - Download: https://zenodo.org/records/18701286/files/lut_sentinel2b_b2to12_3um_dust.mat

- **lut_HLSS30_b1to13_3um_dust.mat** (101 MB)
  - HLS (Harmonized Landsat Sentinel-2) lookup table (also in repository via LFS)
  - Download: https://zenodo.org/records/18701286/files/lut_HLSS30_b1to13_3um_dust.mat

- **lut_modis_b1to7_3um_dust.mat** (537 MB)
  - MODIS lookup table (also in repository via LFS)
  - Download: https://zenodo.org/records/18701286/files/lut_modis_b1to7_3um_dust.mat

- **lut_oli_b1to7_3um_dust.mat** (55 MB)
  - Landsat OLI lookup table (also in repository via LFS)
  - Download: https://zenodo.org/records/18701286/files/lut_oli_b1to7_3um_dust.mat

### Test Imagery
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18704072.svg)](https://doi.org/10.5281/zenodo.18704072)

- **sentinel_r.nc** (1.4 GB)
  - Full spatial resolution (921×1347 pixels) Sentinel-2 reflectance data
  - 9 spectral bands, 2 time steps
  - Download: https://zenodo.org/records/18704072/files/sentinel_r.nc

- **sentinel_r0.nc** (705 MB)
  - Full spatial resolution background reflectance
  - Download: https://zenodo.org/records/18704072/files/sentinel_r0.nc

## Usage

### For Development

All lookup tables and test subsets are included in the repository via Git LFS:

```python
import xarray as xr
import spires

# Sentinel-2 example
r = xr.open_dataset('tests/data/sentinel_r_subset.nc')
r0 = xr.open_dataset('tests/data/sentinel_r0_subset.nc')
lut_s2 = spires.LutInterpolator('tests/data/lut_sentinel2b_b2to12_3um_dust.mat')

# MODIS example
lut_modis = spires.LutInterpolator('tests/data/lut_modis_b1to7_3um_dust.mat')

# Landsat OLI example
lut_oli = spires.LutInterpolator('tests/data/lut_oli_b1to7_3um_dust.mat')

# HLS example
lut_hls = spires.LutInterpolator('tests/data/lut_HLSS30_b1to13_3um_dust.mat')
```

### For Full Tests

To run tests with full-resolution imagery or download LUTs without LFS, get from Zenodo:

```bash
cd tests/data

# Download all LUTs from Zenodo (alternative to LFS)
curl -L -o lut_sentinel2b_b2to12_3um_dust.mat https://zenodo.org/records/18701286/files/lut_sentinel2b_b2to12_3um_dust.mat
curl -L -o lut_HLSS30_b1to13_3um_dust.mat https://zenodo.org/records/18701286/files/lut_HLSS30_b1to13_3um_dust.mat
curl -L -o lut_modis_b1to7_3um_dust.mat https://zenodo.org/records/18701286/files/lut_modis_b1to7_3um_dust.mat
curl -L -o lut_oli_b1to7_3um_dust.mat https://zenodo.org/records/18701286/files/lut_oli_b1to7_3um_dust.mat

# Download legacy MODIS LUT (if needed for compatibility tests)
curl -L -o LUT_MODIS.mat https://zenodo.org/records/18701286/files/LUT_MODIS.mat

# Download full test imagery (large!)
curl -L -o sentinel_r.nc https://zenodo.org/records/18704072/files/sentinel_r.nc
curl -L -o sentinel_r0.nc https://zenodo.org/records/18704072/files/sentinel_r0.nc
```

Or use the provided helper script:
```bash
# Download all large test data files
python scripts/download_test_data.py --all

# Download only specific datasets
python scripts/download_test_data.py --luts
python scripts/download_test_data.py --imagery
```

## CI/CD Behavior

GitHub Actions:
- All lookup tables (Sentinel-2, MODIS, OLI, HLS) are available via Git LFS
- Uses subset imagery files by default (fast tests)
- Downloads `lut_sentinel2b_b2to12_3um_dust.mat` from Zenodo to avoid LFS quota
- Skips tests requiring legacy LUT_MODIS.mat format
- Full-resolution imagery can be downloaded from Zenodo if needed

GitLab CI:
- Configured with `GIT_LFS_SKIP_SMUDGE: "1"` to avoid LFS quota issues
- Downloads required LUTs from Zenodo on-demand
- Uses same test approach as GitHub Actions

## Citation

If you use these datasets in your research, please cite:

```bibtex
@dataset{bair2026spires_luts,
  author       = {Bair, Edward and Dozier, Jeff},
  title        = {{SPIRES} Snow Reflectance Lookup Tables},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18701286},
  url          = {https://doi.org/10.5281/zenodo.18701286}
}

@dataset{griessbaum2026sentinel2_testdata,
  author       = {Griessbaum, Niklas},
  title        = {Sentinel-2 reflectance data for testing the {SpiPy} implementation of the {SPIRES} algorithm},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18704072},
  url          = {https://doi.org/10.5281/zenodo.18704072}
}
```

And the original SPIRES algorithm:
```bibtex
@article{bair2021snow,
  title={Snow Property Inversion From Remote Sensing (SPIReS): A Generalized Multispectral Unmixing Approach With Examples From MODIS and Landsat 8 OLI},
  author={Bair, Edward H and Stillinger, Thomas and Dozier, Jeff},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={59},
  number={9},
  pages={7270--7284},
  year={2021},
  doi={10.1109/TGRS.2020.3040124}
}
```
