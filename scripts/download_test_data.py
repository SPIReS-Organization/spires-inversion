#!/usr/bin/env python3
"""
Download large test data files from Zenodo.

This script downloads full-resolution test data that is too large to store
in the Git repository. Subset files suitable for CI are already in the repo.

Usage:
    python scripts/download_test_data.py --all
    python scripts/download_test_data.py --luts
    python scripts/download_test_data.py --imagery
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path


# Zenodo download URLs
ZENODO_FILES = {
    'luts': {
        'LUT_MODIS.mat': {
            'url': 'https://zenodo.org/records/18701286/files/LUT_MODIS.mat',
            'size_mb': 537,
            'doi': '10.5281/zenodo.18701286'
        },
        'lut_sentinel2b_b2to12_3um_dust.mat': {
            'url': 'https://zenodo.org/records/18701286/files/lut_sentinel2b_b2to12_3um_dust.mat',
            'size_mb': 70,
            'doi': '10.5281/zenodo.18701286',
            'note': 'Also available in repository via Git LFS'
        }
    },
    'imagery': {
        'sentinel_r.nc': {
            'url': 'https://zenodo.org/records/18704072/files/sentinel_r.nc',
            'size_mb': 1400,
            'doi': '10.5281/zenodo.18704072'
        },
        'sentinel_r0.nc': {
            'url': 'https://zenodo.org/records/18704072/files/sentinel_r0.nc',
            'size_mb': 705,
            'doi': '10.5281/zenodo.18704072'
        }
    }
}


def download_file(url, dest_path, filename, size_mb):
    """Download a file with progress reporting."""
    print(f"\nDownloading {filename} ({size_mb} MB)...")
    print(f"  From: {url}")
    print(f"  To: {dest_path}")

    if dest_path.exists():
        response = input(f"  File already exists. Overwrite? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("  Skipped.")
            return

    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print("\n  ✓ Download complete!")

    except Exception as e:
        print(f"\n  ✗ Error downloading file: {e}")
        if dest_path.exists():
            dest_path.unlink()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download large test data files from Zenodo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download all files:
    python scripts/download_test_data.py --all

  Download only lookup tables:
    python scripts/download_test_data.py --luts

  Download only test imagery:
    python scripts/download_test_data.py --imagery

  Download specific file:
    python scripts/download_test_data.py --file LUT_MODIS.mat

For more information, see tests/data/README.md
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Download all large test data files (~2.7 GB)')
    parser.add_argument('--luts', action='store_true',
                        help='Download lookup tables only (~607 MB)')
    parser.add_argument('--imagery', action='store_true',
                        help='Download test imagery only (~2.1 GB)')
    parser.add_argument('--file', type=str,
                        help='Download specific file by name')
    parser.add_argument('--dest', type=str, default='tests/data',
                        help='Destination directory (default: tests/data)')

    args = parser.parse_args()

    # Determine what to download
    if not (args.all or args.luts or args.imagery or args.file):
        parser.print_help()
        print("\nError: Please specify what to download (--all, --luts, --imagery, or --file)")
        sys.exit(1)

    # Setup destination directory
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Build download list
    to_download = {}

    if args.all:
        for category in ZENODO_FILES.values():
            to_download.update(category)
    elif args.file:
        # Find the specific file
        found = False
        for category in ZENODO_FILES.values():
            if args.file in category:
                to_download[args.file] = category[args.file]
                found = True
                break
        if not found:
            print(f"Error: File '{args.file}' not found in catalog")
            print(f"Available files: {', '.join(f for cat in ZENODO_FILES.values() for f in cat.keys())}")
            sys.exit(1)
    else:
        if args.luts:
            to_download.update(ZENODO_FILES['luts'])
        if args.imagery:
            to_download.update(ZENODO_FILES['imagery'])

    # Calculate total size
    total_mb = sum(info['size_mb'] for info in to_download.values())

    print("=" * 70)
    print("spires-inversion Test Data Download")
    print("=" * 70)
    print(f"\nFiles to download: {len(to_download)}")
    print(f"Total size: ~{total_mb} MB ({total_mb/1024:.2f} GB)")
    print(f"Destination: {dest_dir.absolute()}")
    print("\nFiles:")
    for filename, info in to_download.items():
        note = f" - {info['note']}" if 'note' in info else ""
        print(f"  - {filename} ({info['size_mb']} MB){note}")

    response = input("\nProceed with download? [Y/n]: ")
    if response.lower() in ['n', 'no']:
        print("Cancelled.")
        sys.exit(0)

    # Download files
    print("\nStarting downloads...")
    for filename, info in to_download.items():
        dest_path = dest_dir / filename
        download_file(info['url'], dest_path, filename, info['size_mb'])

    print("\n" + "=" * 70)
    print("✓ All downloads complete!")
    print("=" * 70)
    print("\nYou can now run the full test suite:")
    print("  pytest -v")
    print("\nFor more information about these datasets:")
    print(f"  Lookup tables: https://doi.org/10.5281/zenodo.18701286")
    print(f"  Test imagery: https://doi.org/10.5281/zenodo.18704072")


if __name__ == '__main__':
    main()
