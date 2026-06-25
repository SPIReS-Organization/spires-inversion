import numpy as np
import pytest

from spires_inversion.grouping import (
    group_spectra_block,
    group_spectra_rows,
    scatter_group_results,
    scatter_group_results_block,
)


def test_repeated_spectra_group_together():
    targets = np.array([[0.2, 0.3], [0.2, 0.3], [0.8, 0.9]], dtype=np.float64)
    backgrounds = np.array([[0.1, 0.1], [0.1, 0.1], [0.4, 0.4]], dtype=np.float64)
    solar = np.array([30.0, 30.0, 40.0], dtype=np.float64)

    grouped = group_spectra_rows(targets, backgrounds, solar, representative_method="first_pixel")

    assert grouped.n_groups == 2
    np.testing.assert_array_equal(grouped.counts, np.array([2, 1]))
    np.testing.assert_array_equal(grouped.inverse_indices, np.array([0, 0, 1]))


def test_first_representative_uses_first_member():
    targets = np.array([[0.201, 0.301], [0.209, 0.309], [0.7, 0.8]], dtype=np.float64)
    backgrounds = np.array([[0.101, 0.111], [0.109, 0.119], [0.4, 0.5]], dtype=np.float64)
    solar = np.array([30.0, 31.0, 40.0], dtype=np.float64)

    grouped = group_spectra_rows(
        targets,
        backgrounds,
        solar,
        representative_method="first_pixel",
        tolerance=0.05,
    )

    assert grouped.n_groups == 2
    np.testing.assert_allclose(grouped.representative_targets[0], targets[0])
    np.testing.assert_allclose(grouped.representative_backgrounds[0], backgrounds[0])
    np.testing.assert_allclose(grouped.representative_solar_zenith[0], solar[0])


def test_mean_of_pixels_representative_uses_group_mean():
    targets = np.array([[0.201, 0.301], [0.209, 0.309], [0.7, 0.8]], dtype=np.float64)
    backgrounds = np.array([[0.101, 0.111], [0.109, 0.119], [0.4, 0.5]], dtype=np.float64)
    solar = np.array([30.0, 31.0, 40.0], dtype=np.float64)

    grouped = group_spectra_rows(
        targets,
        backgrounds,
        solar,
        representative_method="mean_of_pixels",
        tolerance=0.05,
    )

    assert grouped.n_groups == 2
    np.testing.assert_allclose(grouped.representative_targets[0], targets[:2].mean(axis=0))
    np.testing.assert_allclose(grouped.representative_backgrounds[0], backgrounds[:2].mean(axis=0))
    np.testing.assert_allclose(grouped.representative_solar_zenith[0], solar[:2].mean())


def test_valid_mask_excludes_samples():
    targets = np.array([[0.2, 0.3], [0.2, 0.3], [0.8, 0.9]], dtype=np.float64)
    backgrounds = np.array([[0.1, 0.1], [0.1, 0.1], [0.4, 0.4]], dtype=np.float64)
    valid_mask = np.array([True, False, True])

    grouped = group_spectra_rows(targets, backgrounds, valid_mask=valid_mask)

    assert grouped.n_valid == 2
    np.testing.assert_array_equal(grouped.valid_flat_indices, np.array([0, 2]))


def test_non_finite_inputs_are_excluded():
    targets = np.array([[0.2, 0.3], [np.nan, 0.3], [0.8, 0.9]], dtype=np.float64)
    backgrounds = np.array([[0.1, 0.1], [0.1, 0.1], [np.inf, 0.4]], dtype=np.float64)
    solar = np.array([30.0, 40.0, np.nan], dtype=np.float64)

    grouped = group_spectra_rows(targets, backgrounds, solar)

    assert grouped.n_valid == 1
    np.testing.assert_array_equal(grouped.valid_flat_indices, np.array([0]))
    np.testing.assert_allclose(grouped.representative_targets, np.array([[0.2, 0.3]]))


def test_scatter_group_results_preserves_invalid_nan_fill():
    targets = np.array([[0.2, 0.3], [np.nan, 0.3], [0.8, 0.9]], dtype=np.float64)
    backgrounds = np.array([[0.1, 0.1], [0.1, 0.1], [0.4, 0.4]], dtype=np.float64)

    grouped = group_spectra_rows(targets, backgrounds)
    scattered = scatter_group_results(grouped, np.array([[1.0, 2.0], [3.0, 4.0]]))

    np.testing.assert_allclose(scattered[0], np.array([1.0, 2.0]))
    assert np.isnan(scattered[1]).all()
    np.testing.assert_allclose(scattered[2], np.array([3.0, 4.0]))


def test_scatter_group_results_block_preserves_block_shape():
    targets = np.array(
        [
            [[0.2, 0.3], [0.2, 0.3]],
            [[0.8, 0.9], [0.5, 0.6]],
        ],
        dtype=np.float64,
    )
    backgrounds = np.full_like(targets, 0.1)

    grouped = group_spectra_block(targets, backgrounds)
    group_results = np.column_stack([np.arange(grouped.n_groups), np.arange(grouped.n_groups) + 10])
    scattered = scatter_group_results_block(grouped, group_results)

    assert scattered.shape == (2, 2, 2)
    np.testing.assert_allclose(scattered[0, 0], scattered[0, 1])


def test_group_spectra_block_accepts_broadcastable_inputs():
    targets = np.ones((2, 3, 4, 2), dtype=np.float64)
    backgrounds = np.ones((3, 4, 2), dtype=np.float64) * 0.1
    solar = np.full((1, 3, 4), 35.0, dtype=np.float64)
    valid_mask = np.ones((3, 4), dtype=bool)

    grouped = group_spectra_block(
        targets,
        backgrounds,
        solar,
        valid_mask=valid_mask,
    )

    assert grouped.original_shape == targets.shape
    assert grouped.n_valid == 24
    assert grouped.n_groups == 1


def test_invalid_representative_method_raises():
    with pytest.raises(ValueError, match="representative_method"):
        group_spectra_rows(
            np.ones((1, 2)),
            np.ones((1, 2)),
            representative_method="median",
        )


@pytest.mark.parametrize("old_name", ["first", "chunk_bin_mean"])
def test_old_representative_method_names_raise(old_name):
    with pytest.raises(ValueError, match="representative_method"):
        group_spectra_rows(
            np.ones((1, 2)),
            np.ones((1, 2)),
            representative_method=old_name,
        )
