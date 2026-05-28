"""Verify that the C++ inversion releases the GIL so multiple Python
threads can run inversions in parallel within a single process."""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import spires
from spires.invert import speedy_invert_array2d


interpolator = spires.LutInterpolator(
    lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')

spectrum_target = np.array(
    [0.3424, 0.366, 0.3624, 0.38932347, 0.41624767, 0.39567757,
     0.07043362, 0.06267947, 0.3792])
spectrum_background = np.array(
    [0.0182, 0.0265, 0.0283, 0.05606749, 0.09543234, 0.12036866,
     0.12491679, 0.07888655, 0.1406])
solar_angle = 55.73733298
expected_per_pixel = np.array(
    [4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02])


def _make_chunk(ny=8, nx=8):
    n_bands = spectrum_target.size
    targets = np.broadcast_to(spectrum_target, (ny, nx, n_bands)).copy()
    backgrounds = np.broadcast_to(spectrum_background, (ny, nx, n_bands)).copy()
    angles = np.full((ny, nx), solar_angle)
    return targets, backgrounds, angles


def _invert_one(_):
    targets, backgrounds, angles = _make_chunk(ny=20, nx=20)
    return speedy_invert_array2d(
        spectra_targets=targets,
        spectra_backgrounds=backgrounds,
        obs_solar_angles=angles,
        interpolator=interpolator,
        algorithm=1,
        max_eval=200)


def test_thread_pool_results_match_serial():
    """Same numeric output whether run serially or across threads."""
    serial = [_invert_one(i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        threaded = list(ex.map(_invert_one, range(4)))
    for s, t in zip(serial, threaded):
        np.testing.assert_allclose(s, t, rtol=1e-4)
    np.testing.assert_allclose(
        threaded[0][0, 0], expected_per_pixel, rtol=1e-4)


def test_threads_achieve_parallel_speedup():
    """Wall-clock speedup proves the GIL is released during inversion.

    If the GIL were held, threaded time would equal serial time. We require
    a modest speedup (>1.4x with 4 threads) to leave headroom for noisy CI.
    """
    n_jobs = 8

    t0 = time.perf_counter()
    for i in range(n_jobs):
        _invert_one(i)
    serial_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_invert_one, range(n_jobs)))
    threaded_time = time.perf_counter() - t0

    speedup = serial_time / threaded_time
    assert speedup > 1.4, (
        f"Expected >1.4x speedup with 4 threads (GIL released), "
        f"got {speedup:.2f}x (serial={serial_time:.2f}s, "
        f"threaded={threaded_time:.2f}s)")
