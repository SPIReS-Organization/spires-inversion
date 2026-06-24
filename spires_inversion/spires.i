%module core

%{
  #define SWIG_FILE_WITH_INIT  /* To import_array() below */
  #include "spires.h"
  #include <vector>
  #include <string>
  #include <exception>

  /* RAII guard that releases the Python GIL on construction and restores it on
     destruction. Exception-safe: if the wrapped C++ code throws (e.g. NLopt),
     the destructor reacquires the GIL before the SWIG exception machinery
     converts it to a Python exception. */
  class SpiresGILRelease {
      PyThreadState* _save;
  public:
      SpiresGILRelease() : _save(PyEval_SaveThread()) {}
      ~SpiresGILRelease() { PyEval_RestoreThread(_save); }
      SpiresGILRelease(const SpiresGILRelease&) = delete;
      SpiresGILRelease& operator=(const SpiresGILRelease&) = delete;
  };
%}

%include "numpy.i"

%init %{
    import_array();
%}

%include "std_vector.i"
namespace std {
    %template(DoubleVector) vector<double>;
}


%apply (double* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4) { 
    (double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes)
};

%apply(double* IN_ARRAY3, int DIM1, int DIM2, int DIM3){
    (double* spectra_backgrounds, int n_background_y, int n_background_x, int n_bands_backgrounds),
    (double* spectra_targets, int n_target_y, int n_target_x, int n_bands_target)
}


%apply (double* IN_ARRAY1, int DIM1) { 
    (double* spectrum_background, int len_background),
    (double* spectrum_target, int len_target),
    (double* spectrum_shade, int len_shade),
    (double* grain_sizes, int len_grain_sizes),
    (double* dust_concentrations, int len_dust_concentrations),
    (double* solar_angles, int len_solar_angles),
    (double* bands, int len_bands),
    (double* obs_solar_angles, int n_obs_solar_angles)   
}

%apply (double* INPLACE_ARRAY1, int DIM1) {
    (double* x, int len_x)
}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {
       (double* spectra_backgrounds, int n_obs_backgrounds, int n_bands_backgrounds),
       (double* spectra_targets, int n_obs_target, int n_bands_target),
       (double* obs_solar_angles, int n_obs_solar_y, int n_obs_solar_x)
}

%apply(double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double* results, int n_obs, int n_results)
 }

%apply(double* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {
    (double* results, int n_y, int n_x, int n_results)
}

/* Output typemap for `double*` returns from interpolate_all_array.
   Length is taken from `arg2` (n_bands), which is set by the IN_ARRAY4
   typemap on the `lut` argument. The returned buffer was allocated with
   `new double[n_bands]` in C++, so we transfer ownership to NumPy by
   wrapping it in a PyCapsule with a `delete[]` destructor and using that
   capsule as the array's base — this guarantees the buffer is freed when
   the NumPy array is deallocated, fixing the previous leak. */
%typemap(out) double* {
    npy_intp dims[1] = { (npy_intp)arg2 };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)$1);
    if (!arr) {
        delete[] $1;
        SWIG_fail;
    }
    PyObject* capsule = PyCapsule_New((void*)$1, NULL,
        [](PyObject* cap) {
            void* p = PyCapsule_GetPointer(cap, NULL);
            delete[] static_cast<double*>(p);
        });
    if (!capsule) {
        Py_DECREF(arr);
        delete[] $1;
        SWIG_fail;
    }
    if (PyArray_SetBaseObject((PyArrayObject*)arr, capsule) < 0) {
        Py_DECREF(arr);  // PyArray_SetBaseObject steals capsule ref on failure? No — on failure we still own capsule.
        Py_DECREF(capsule);
        SWIG_fail;
    }
    $result = arr;
}

/* Release the GIL during the long-running NLopt-driven inversions so that
   Python threads (e.g. Dask threaded workers) can run other work in parallel
   on the same process. Argument extraction (input typemaps) runs before the
   try-block with the GIL held; only the bare C++ call ($action) is GIL-free.
   Output typemaps run after the guard is destroyed, with the GIL re-held. */
%exception invert {
    try {
        SpiresGILRelease _gil;
        $action
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%exception invert_array1d {
    try {
        SpiresGILRelease _gil;
        $action
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%exception invert_array2d {
    try {
        SpiresGILRelease _gil;
        $action
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}


%include spires.h