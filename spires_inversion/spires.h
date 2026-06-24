#include <vector>


// Function declaration for interpolation

double interpolate_idx(double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes, 
                       int band_idx, double solar_angle_idx, double dust_concentration_idx, double grain_size_idx);


double interpolate(double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes,
                   double* lut_bands, int len_lut_bands,
                   double* lut_solar_angles, int len_lut_solar_angles,
                   double* lut_dust_concentrations, int len_lut_dust_concentrations,
                   double* lut_grain_sizes, int len_lut_grain_sizes,
                   int band,
                   double solar_angle, 
                   double dust_concentration, 
                   double grain_size);

std::vector<double> interpolate_all_idx(double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes,
                                        double solar_angle, double dust, double grain_size);

std::vector<double> interpolate_all(double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes, 
                                    double* lut_bands, int len_lut_bands,
                                    double* lut_solar_angles, int len_lut_solar_angles,
                                    double* lut_dust_concentrations, int len_lut_dust_concentrations,
                                    double* lut_grain_sizes, int len_lut_grain_sizes,
                                    double solar_angle, 
                                    double dust_size, 
                                    double grain_size);


double* interpolate_all_array(double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes,
                              double* lut_bands, int len_lut_bands,
                              double* lut_solar_angles, int len_lut_solar_angles,
                              double* lut_dust_concentrations, int len_lut_dust_concentrations,
                              double* lut_grain_sizes, int len_lut_grain_sizes,
                              double solar_angle,
                              double dust_concentration,
                              double grain_size);

double spectrum_difference(const std::vector<double>& x,
                           double* spectrum_background, int len_background, 
                           double* spectrum_target, int len_target,
                           double* spectrum_shade, int len_shade,
                           double solar_angle,       
                           double* lut_bands, int len_lut_bands,
                           double* lut_solar_angles, int len_lut_solar_angles,
                           double* lut_dust_concentrations, int len_lut_dust_concentrations,
                           double* lut_grain_sizes, int len_lut_grain_sizes,                     
                           double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes);

double spectrum_difference_scaled(const std::vector<double> &x,
                                  double* spectrum_background, int len_background,
                                  double* spectrum_target, int len_target,
                                  double* spectrum_shade, int len_shade,
                                  double solar_angle,
                                  double* lut_bands, int len_lut_bands,
                                  double* lut_solar_angles, int len_lut_solar_angles,
                                  double* lut_dust_concentrations, int len_lut_dust_concentrations,
                                  double* lut_grain_sizes, int len_lut_grain_sizes,
                                  double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes);


double spectrum_difference_softmax(const std::vector<double>& z,
                                   double* spectrum_background, int len_background,
                                   double* spectrum_target, int len_target,
                                   double* spectrum_shade, int len_shade,
                                   double solar_angle,
                                   double* lut_bands, int len_lut_bands,
                                   double* lut_solar_angles, int len_lut_solar_angles,
                                   double* lut_dust_concentrations, int len_lut_dust_concentrations,
                                   double* lut_grain_sizes, int len_lut_grain_sizes,
                                   double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes);


double spectrum_difference_hybrid(const std::vector<double>& y,
                                  double* spectrum_background, int len_background,
                                  double* spectrum_target, int len_target,
                                  double* spectrum_shade, int len_shade,
                                  double solar_angle,
                                  double* lut_bands, int len_lut_bands,
                                  double* lut_solar_angles, int len_lut_solar_angles,
                                  double* lut_dust_concentrations, int len_lut_dust_concentrations,
                                  double* lut_grain_sizes, int len_lut_grain_sizes,
                                  double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes);


std::vector<double> z_to_x(const std::vector<double>& z,
                           double dust_min, double dust_max,
                           double grain_min, double grain_max);


std::vector<double> x_to_z(const std::vector<double>& x,
                           double dust_min, double dust_max,
                           double grain_min, double grain_max);


std::vector<double> y_to_x_hybrid(const std::vector<double>& y);
std::vector<double> x_to_y_hybrid(const std::vector<double>& x);


std::vector<double>  invert(double* spectrum_background, int len_background,
                           double* spectrum_target, int len_target,
                           double* spectrum_shade, int len_shade,
                           double solar_angle,
                           double* lut_bands, int len_lut_bands,
                           double* lut_solar_angles, int len_lut_solar_angles,
                           double* lut_dust_concentrations, int len_lut_dust_concentrations,
                           double* lut_grain_sizes, int len_lut_grain_sizes,
                           double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes,                           
                           int max_eval,
                           std::vector<double> x0,
                           int algorithm);



void invert_array1d(double* spectra_backgrounds, int n_obs_backgrounds, int n_bands_backgrounds,
                    double* spectra_targets, int n_obs_target, int n_bands_target,
                    double* spectrum_shade, int len_shade,
                    double* obs_solar_angles, int n_obs_solar_angles,
                    double* lut_bands, int len_lut_bands,
                    double* lut_solar_angles, int len_lut_solar_angles,
                    double* lut_dust_concentrations, int len_lut_dust_concentrations,
                    double* lut_grain_sizes, int len_lut_grain_sizes,
                    double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes,
                    double* results, int n_obs, int n_results,
                    int max_eval,
                    std::vector<double> x0,
                    int algorithm);


void invert_array2d(double* spectra_backgrounds, int n_background_y, int n_background_x, int n_bands_backgrounds,
                    double* spectra_targets, int n_target_y, int n_target_x, int n_bands_target,
                    double* spectrum_shade, int len_shade,
                    double* obs_solar_angles, int n_obs_solar_y, int n_obs_solar_x,
                    double* lut_bands, int len_lut_bands,
                    double* lut_solar_angles, int len_lut_solar_angles,
                    double* lut_dust_concentrations, int len_lut_dust_concentrations,
                    double* lut_grain_sizes, int len_lut_grain_sizes,
                    double* lut_reflectances, int n_lut_bands, int n_lut_solar_angles, int n_lut_dust_concentrations, int n_lut_grain_sizes,
                    double* results, int n_y, int n_x, int n_results,
                    int max_eval,
                    std::vector<double> x0,
                    int algorithm);