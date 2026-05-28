#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <nlopt.hpp>
#include "spires.h"


// ----------------------------------------------------------------------------
// Index-space helpers
// ----------------------------------------------------------------------------

static inline double linearInterpolate(double y1, double y2, double x, double x1, double x2) {
    return y1 + (y2 - y1) * ((x - x1) / (x2 - x1));
}


double interpolate_idx(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                       int band_idx, double solar_angle_idx, double dust_concentration_idx, double grain_size_idx) {

    // Half-open contract: valid index ranges are [0, n_*).
    if (band_idx < 0 || band_idx >= n_bands ||
        solar_angle_idx < 0 || solar_angle_idx > n_solar_angles - 1 ||
        dust_concentration_idx < 0 || dust_concentration_idx > n_dust_concentrations - 1 ||
        grain_size_idx < 0 || grain_size_idx > n_grain_sizes - 1) {
        std::cerr << "Error: Coordinates out of bounds" << std::endl;
        return -1;
    }

    // Select the 3D cube for this band; we interpolate in solar/dust/grain only.
    int start_idx = band_idx * (n_solar_angles * n_dust_concentrations * n_grain_sizes);
    double* cube = lut + start_idx;

    int iz1 = static_cast<int>(solar_angle_idx);
    int id1 = static_cast<int>(dust_concentration_idx);
    int iw1 = static_cast<int>(grain_size_idx);
    // Clamp ceiling indices so a coord exactly at the upper bound doesn't read
    // past the end. When clamped, the "interpolation" between iz1==iz2 reduces
    // to v at iz1, which is the desired behavior.
    int iz2 = std::min(iz1 + 1, n_solar_angles - 1);
    int id2 = std::min(id1 + 1, n_dust_concentrations - 1);
    int iw2 = std::min(iw1 + 1, n_grain_sizes - 1);

    double v000 = cube[n_grain_sizes * (id1 + iz1 * n_dust_concentrations) + iw1];
    double v001 = cube[n_grain_sizes * (id1 + iz1 * n_dust_concentrations) + iw2];
    double v010 = cube[n_grain_sizes * (id2 + iz1 * n_dust_concentrations) + iw1];
    double v011 = cube[n_grain_sizes * (id2 + iz1 * n_dust_concentrations) + iw2];
    double v100 = cube[n_grain_sizes * (id1 + iz2 * n_dust_concentrations) + iw1];
    double v101 = cube[n_grain_sizes * (id1 + iz2 * n_dust_concentrations) + iw2];
    double v110 = cube[n_grain_sizes * (id2 + iz2 * n_dust_concentrations) + iw1];
    double v111 = cube[n_grain_sizes * (id2 + iz2 * n_dust_concentrations) + iw2];

    return linearInterpolate(
        linearInterpolate(
            linearInterpolate(v000, v001, grain_size_idx, iw1, iw2),
            linearInterpolate(v010, v011, grain_size_idx, iw1, iw2),
            dust_concentration_idx, id1, id2
        ),
        linearInterpolate(
            linearInterpolate(v100, v101, grain_size_idx, iw1, iw2),
            linearInterpolate(v110, v111, grain_size_idx, iw1, iw2),
            dust_concentration_idx, id1, id2
        ),
        solar_angle_idx, iz1, iz2
    );
}


std::vector<double> interpolate_all_idx(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                                        double solar_angle_idx, double dust_concentration_idx, double grain_size_idx) {
    std::vector<double> spectrum(n_bands);
    for (int band_idx = 0; band_idx < n_bands; band_idx++) {
        spectrum[band_idx] = interpolate_idx(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                                             band_idx, solar_angle_idx, dust_concentration_idx, grain_size_idx);
    }
    return spectrum;
}


// In-place variant: writes into a caller-supplied buffer to avoid the heap
// allocation in the optimizer's hot path. Caller guarantees |out| >= n_bands.
static inline void interpolate_all_idx_into(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                                            double solar_angle_idx, double dust_concentration_idx, double grain_size_idx,
                                            double* out) {
    for (int band_idx = 0; band_idx < n_bands; band_idx++) {
        out[band_idx] = interpolate_idx(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                                        band_idx, solar_angle_idx, dust_concentration_idx, grain_size_idx);
    }
}


double get_idx_linspace(double value, double* coordinates, int len_coordinates) {
    return (value - coordinates[0]) / (coordinates[len_coordinates - 1] - coordinates[0]) * (len_coordinates - 1);
}


double get_idx(double value, double* coordinates, int len_coordinates) {
    // Preserves original semantics: `left_index` is the smallest index where
    // coordinates[left_index] >= value (or len-1 if value exceeds the array).
    // Interpolation then uses the [left_index, left_index+1] bracket, which
    // means the returned float index can lie below `left_index` for values
    // between two coordinates — callers (interpolate_idx) handle this by
    // taking the floor.
    //
    // Bug fix: when value is at/above the upper bound, the original code
    // would set right_index = len_coordinates and dereference past the end.
    // Clamp left_index to len-2 so right_index = left_index+1 stays in range.
    size_t left_index = 0;
    while (left_index < static_cast<size_t>(len_coordinates - 1) && coordinates[left_index] < value) {
        left_index++;
    }
    if (left_index >= static_cast<size_t>(len_coordinates - 1)) {
        left_index = len_coordinates - 2;
        if (len_coordinates < 2) return 0.0;
    }
    size_t right_index = left_index + 1;
    double left_coord = coordinates[left_index];
    double right_coord = coordinates[right_index];
    if (left_coord == value) {
        return static_cast<double>(left_index);
    }
    double interpolation_factor = (value - left_coord) / (right_coord - left_coord);
    return left_index + interpolation_factor;
}


double interpolate(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                   double* bands, int len_bands,
                   double* solar_angles, int len_solar_angles,
                   double* dust_concentrations, int len_dust_concentrations,
                   double* grain_sizes, int len_grain_sizes,
                   int band,
                   double solar_angle,
                   double dust_concentration,
                   double grain_size) {
    double solar_angle_idx = get_idx(solar_angle, solar_angles, len_solar_angles);
    double dust_concentration_idx = get_idx(dust_concentration, dust_concentrations, len_dust_concentrations);
    double grain_size_idx = get_idx(grain_size, grain_sizes, len_grain_sizes);
    int band_idx = band - 1;
    return interpolate_idx(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                           band_idx, solar_angle_idx, dust_concentration_idx, grain_size_idx);
}


std::vector<double> interpolate_all(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                                    double* bands, int len_bands,
                                    double* solar_angles, int len_solar_angles,
                                    double* dust_concentrations, int len_dust_concentrations,
                                    double* grain_sizes, int len_grain_sizes,
                                    double solar_angle,
                                    double dust_concentration,
                                    double grain_size) {
    double solar_angle_idx = get_idx(solar_angle, solar_angles, len_solar_angles);
    double dust_concentration_idx = get_idx(dust_concentration, dust_concentrations, len_dust_concentrations);
    double grain_size_idx = get_idx(grain_size, grain_sizes, len_grain_sizes);
    return interpolate_all_idx(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                               solar_angle_idx, dust_concentration_idx, grain_size_idx);
}


double* interpolate_all_array(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                              double* bands, int len_bands,
                              double* solar_angles, int len_solar_angles,
                              double* dust_concentrations, int len_dust_concentrations,
                              double* grain_sizes, int len_grain_sizes,
                              double solar_angle,
                              double dust_concentration,
                              double grain_size) {
    // Allocates n_bands doubles; ownership transfers to caller via SWIG typemap.
    double solar_angle_idx = get_idx(solar_angle, solar_angles, len_solar_angles);
    double dust_concentration_idx = get_idx(dust_concentration, dust_concentrations, len_dust_concentrations);
    double grain_size_idx = get_idx(grain_size, grain_sizes, len_grain_sizes);

    double* spectrum = new double[n_bands];
    interpolate_all_idx_into(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                             solar_angle_idx, dust_concentration_idx, grain_size_idx, spectrum);
    return spectrum;
}


// ----------------------------------------------------------------------------
// Spectrum-difference cost (fused, allocation-free)
// ----------------------------------------------------------------------------

double spectrum_difference(const std::vector<double>& x,
                           double* spectrum_background, int len_background,
                           double* spectrum_target, int len_target,
                           double* spectrum_shade, int len_shade,
                           double solar_angle,
                           double* bands, int len_bands,
                           double* solar_angles, int len_solar_angles,
                           double* dust_concentrations, int len_dust_concentrations,
                           double* grain_sizes, int len_grain_sizes,
                           double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes) {
    /*
    Euclidean distance between modeled and measured reflectance.
    x = [f_sca, f_shade, dust, grain_size]

    The previous implementation built four std::vector<double> buffers per call
    via the Spectrum class. This fused loop is allocation-free apart from the
    per-call interpolated-spectrum buffer.
    */
    double f_sca = x[0];
    double f_shade = x[1];
    double dust = x[2];
    double grain_size = x[3];
    double f_bg = 1.0 - f_sca - f_shade;

    double solar_angle_idx = get_idx(solar_angle, solar_angles, len_solar_angles);
    double dust_idx = get_idx(dust, dust_concentrations, len_dust_concentrations);
    double grain_idx = get_idx(grain_size, grain_sizes, len_grain_sizes);

    double diff_sq = 0.0;
    for (int i = 0; i < len_target; ++i) {
        double model_pure = interpolate_idx(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                                            i, solar_angle_idx, dust_idx, grain_idx);
        double model = model_pure * f_sca + spectrum_shade[i] * f_shade + spectrum_background[i] * f_bg;
        double d = spectrum_target[i] - model;
        diff_sq += d * d;
    }
    return std::sqrt(diff_sq);
}


double index_to_value(double value, double* coords, int len_coords) {
    // value in [0, 1] mapped to coords[0] ... coords[len-1].
    // Previous version used `idx = value * len_coords` which read past the end
    // when value == 1.0. Corrected to `value * (len_coords - 1)` with a clamp.
    if (value < 0.0) value = 0.0;
    if (value > 1.0) value = 1.0;
    double idx = value * (len_coords - 1);
    int l_idx = static_cast<int>(idx);
    if (l_idx >= len_coords - 1) {
        return coords[len_coords - 1];
    }
    int r_idx = l_idx + 1;
    double dist = idx - l_idx;
    return coords[l_idx] + dist * (coords[r_idx] - coords[l_idx]);
}


double spectrum_difference_scaled(const std::vector<double>& x,
                                  double* spectrum_background, int len_background,
                                  double* spectrum_target, int len_target,
                                  double* spectrum_shade, int len_shade,
                                  double solar_angle,
                                  double* bands, int len_bands,
                                  double* solar_angles, int len_solar_angles,
                                  double* dust_concentrations, int len_dust_concentrations,
                                  double* grain_sizes, int len_grain_sizes,
                                  double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes) {

    double dust_scaled = index_to_value(x[2], dust_concentrations, len_dust_concentrations);
    double grain_scaled = index_to_value(x[3], grain_sizes, len_grain_sizes);

    std::vector<double> x_scaled = {x[0], x[1], dust_scaled, grain_scaled};

    return spectrum_difference(x_scaled,
                               spectrum_background, len_background,
                               spectrum_target, len_target,
                               spectrum_shade, len_shade,
                               solar_angle,
                               bands, len_bands,
                               solar_angles, len_solar_angles,
                               dust_concentrations, len_dust_concentrations,
                               grain_sizes, len_grain_sizes,
                               lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes);
}


// ----------------------------------------------------------------------------
// NLopt objective wrapping
// ----------------------------------------------------------------------------

struct ObjectiveData {
    double* lut;
    int n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes;
    double* spectrum_background; int len_background;
    double* spectrum_target;     int len_target;
    double* spectrum_shade;      int len_shade;
    double solar_angle;
    double* bands;               int len_bands;
    double* solar_angles;        int len_solar_angles;
    double* dust_concentrations; int len_dust_concentrations;
    double* grain_sizes;         int len_grain_sizes;
};


static double spectrum_difference_wrapper(const std::vector<double>& x, std::vector<double>& /*grad*/, void* data) {
    ObjectiveData* d = reinterpret_cast<ObjectiveData*>(data);
    return spectrum_difference(x,
                               d->spectrum_background, d->len_background,
                               d->spectrum_target, d->len_target,
                               d->spectrum_shade, d->len_shade,
                               d->solar_angle,
                               d->bands, d->len_bands,
                               d->solar_angles, d->len_solar_angles,
                               d->dust_concentrations, d->len_dust_concentrations,
                               d->grain_sizes, d->len_grain_sizes,
                               d->lut, d->n_bands, d->n_solar_angles, d->n_dust_concentrations, d->n_grain_sizes);
}


static double constraint(const std::vector<double>& x, std::vector<double>& /*grad*/, void* /*data*/) {
    // f_sca + f_shade <= 1, written as f_sca + f_shade - 1 <= 0
    return x[0] + x[1] - 1;
}


static bool spectrum_has_nan(double* spectrum, int len) {
    for (int n = 0; n < len; n++) {
        if (std::isnan(spectrum[n])) return true;
    }
    return false;
}


std::vector<double> invert(double* spectrum_background, int len_background,
                           double* spectrum_target, int len_target,
                           double* spectrum_shade, int len_shade,
                           double solar_angle,
                           double* bands, int len_bands,
                           double* solar_angles, int len_solar_angles,
                           double* dust_concentrations, int len_dust_concentrations,
                           double* grain_sizes, int len_grain_sizes,
                           double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                           int max_eval,
                           std::vector<double> x0,
                           int algorithm) {

    if (spectrum_has_nan(spectrum_target, len_target)) {
        return std::vector<double>(4, std::nan(""));
    }

    ObjectiveData obj_data{
        lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
        spectrum_background, len_background,
        spectrum_target, len_target,
        spectrum_shade, len_shade,
        solar_angle,
        bands, len_bands,
        solar_angles, len_solar_angles,
        dust_concentrations, len_dust_concentrations,
        grain_sizes, len_grain_sizes,
    };

    nlopt::opt opt;
    bool constrained_algorithm;

    switch (algorithm) {
    case 1: {
        // LN_COBYLA: derivative-free, supports inequality constraints.
        opt = nlopt::opt(nlopt::LN_COBYLA, 4);
        constrained_algorithm = true;
        std::vector<double> rhobeg = {0.1, 0.1, 100, 100};
        opt.set_initial_step(rhobeg);
        break;
    }
    case 2:
        // LN_NELDERMEAD: derivative-free, no constraint support.
        opt = nlopt::opt(nlopt::LN_NELDERMEAD, 4);
        constrained_algorithm = false;
        break;
    case 3:
        // LD_SLSQP: gradient-based; we don't supply analytic gradients, so
        // NLopt falls back to finite differences. Currently underperforms
        // COBYLA — kept for parity with the Python interface.
        opt = nlopt::opt(nlopt::LD_SLSQP, 4);
        constrained_algorithm = true;
        break;
    default:
        throw std::invalid_argument("invert: unknown algorithm code (must be 1=COBYLA, 2=NELDERMEAD, 3=SLSQP)");
    }

    if (constrained_algorithm) {
        opt.add_inequality_constraint(constraint, &obj_data);
    }
    opt.set_min_objective(spectrum_difference_wrapper, &obj_data);
    opt.set_maxeval(max_eval);

    double min_dust_concentration = dust_concentrations[0];
    double max_dust_concentration = dust_concentrations[len_dust_concentrations - 1];
    double min_grain_size = grain_sizes[0];
    double max_grain_size = grain_sizes[len_grain_sizes - 1];

    std::vector<double> lower_bounds = {0.0, 0.0, min_dust_concentration, min_grain_size};
    std::vector<double> upper_bounds = {1.0, 1.0, max_dust_concentration, max_grain_size};
    opt.set_lower_bounds(lower_bounds);
    opt.set_upper_bounds(upper_bounds);

    opt.set_ftol_abs(1e-4);
    opt.set_xtol_rel(1e-2);

    double minf;
    std::vector<double> x = x0;
    opt.optimize(x, minf);
    return x;
}


void invert_array1d(double* spectra_backgrounds, int n_obs_backgrounds, int n_bands_backgrounds,
                    double* spectra_targets, int n_obs_target, int n_bands_target,
                    double* spectrum_shade, int len_shade,
                    double* obs_solar_angles, int n_obs_solar_angles,
                    double* bands, int len_bands,
                    double* solar_angles, int len_solar_angles,
                    double* dust_concentrations, int len_dust_concentrations,
                    double* grain_sizes, int len_grain_sizes,
                    double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                    double* results, int n_obs, int n_results,
                    int max_eval,
                    std::vector<double> x0,
                    int algorithm) {
    for (int obs = 0; obs < n_obs_backgrounds; obs++) {
        int n = obs * n_bands_backgrounds;
        std::vector<double> x = invert(&spectra_backgrounds[n], len_bands,
                                       &spectra_targets[n], len_bands,
                                       spectrum_shade, len_shade,
                                       obs_solar_angles[obs],
                                       bands, len_bands,
                                       solar_angles, len_solar_angles,
                                       dust_concentrations, len_dust_concentrations,
                                       grain_sizes, len_grain_sizes,
                                       lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                                       max_eval, x0, algorithm);
        for (size_t i = 0; i < x.size(); ++i) {
            results[obs * n_results + i] = x[i];
        }
    }
}


void invert_array2d(double* spectra_backgrounds, int n_background_y, int n_background_x, int n_bands_backgrounds,
                    double* spectra_targets, int n_target_y, int n_target_x, int n_bands_target,
                    double* spectrum_shade, int len_shade,
                    double* obs_solar_angles, int n_obs_solar_y, int n_obs_solar_x,
                    double* bands, int len_bands,
                    double* solar_angles, int len_solar_angles,
                    double* dust_concentrations, int len_dust_concentrations,
                    double* grain_sizes, int len_grain_sizes,
                    double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                    double* results, int n_y, int n_x, int n_results,
                    int max_eval,
                    std::vector<double> x0,
                    int algorithm) {
    for (int y = 0; y < n_target_y; y++) {
        for (int x = 0; x < n_target_x; x++) {
            int obs = x + y * n_target_x;
            int n = obs * n_bands_target;
            std::vector<double> result = invert(&spectra_backgrounds[n], len_bands,
                                                &spectra_targets[n], len_bands,
                                                spectrum_shade, len_shade,
                                                obs_solar_angles[obs],
                                                bands, len_bands,
                                                solar_angles, len_solar_angles,
                                                dust_concentrations, len_dust_concentrations,
                                                grain_sizes, len_grain_sizes,
                                                lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                                                max_eval, x0, algorithm);
            for (size_t i = 0; i < result.size(); ++i) {
                results[obs * n_results + i] = result[i];
            }
        }
    }
}
