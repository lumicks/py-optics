# Changelog

## v0.7.0 | TBD

### Changes

* Bumped minimum version of Python to 3.10
* Bumped maximum tested version of Python to 3.13
* Removed Lebedev-Laikov code in favor of Scipy's `lebedev_rule` function.
* Removed chirped z-transform code in favor of Scipy's `CZT` class.
* Renamed `integration_orders` to `integration_order`
* The signature of the callback used to calculate the field on the back focal plane has changed (see
  examples for new type).

### Improvements

* Added Pytest for CZT-based far-field transform
* Added Gauss-Legendre and Clenshaw-Curtis integration methods for force calculations (in case orders higher than those supported by Lebedev-Laikov are required)
* Changed the default integration order to twice the number of Mie modes for the Lebedev-Laikov integration scheme when no integration order is given.
* Added the electromagnetic field distribution for magnetic dipoles at arbitrary orientations
* Added a method to determine the minimal sampling order of the back focal plane to `objective.Objective`
* Added a Gauss-Legendre-style integration scheme to `mathutils.integration` for integration over a disk or annulus. The application is for the back focal plane, and results in faster convergence compared to equidistant sampling, when using non-czt-based calculation methods to obtain a focus.
* Added `psf.czt` and `psf.quad`, two modules that contain functions to calculate the Point Spread Function (PSF) of arbitrary fields on the back focal plane of an objective. They use the new callback signature and avoid private implementations of what is already calculated by the `Objective` class. The names of the modules make the method of calculation more explicit, namely either the Chirped-Z Tranform (CZT) is used, or a 2D quadrature method is used. The `psf.quad` module supports a new integration method called "peirce" that is dedicated to integration over a circular domain, such as the back focal plane. It typically converges faster than the equidistant ("czt") methods, but the actual CZT implementations could still beat quadrature on raw speed.

### Deprecations

* Deprecated the modules `psf.fast` and `psf.direct`, and the functions `psf.fast_psf` and `psf.fast_gauss`. The new functions (see above) are nearly the same as the deprecated versions, but take an `Objective` class instance instead of four parameters for `n_bfp`, `n_medium`, `NA` and `focal_length`, and use the new callback signature. 

## v0.6.0 | 2024-11-15

### Improvements

* Speed up trapping calculations by making loops multi-threaded
* Update Trapping forces example with closures

## v0.5.0 | 2024-11-05

### Changes
* Bumped minimum version of Python to 3.9, to easily accomodate typing
* Bumped maximum tested version of Python to 3.12

### Improvements

* Added closures to prevent recalculating fixed, but computationally intense, data for force calculations. See `trapping.force_factory`.
* Speed up trapping calculations by rewriting code such that it can be compiled by Numba
* Added functionality to shift the coordinate system for `fast_psf()`
* Added Pytest tests for example notebooks
* Added Pytest test for czt equivalence between FFT and czt for specific cases

### Bug fixes
* Fixed broken czt transform for array inputs
* Fixed bug where the electric field was not correct as a function of z in `psf.reference.focused_dipole_ref`
* Fixed bug where the Z grid points were not returned even with `return_grid` being True in `psf.reference.focused_dipole_ref`

## v0.4.0 | 2023-08-19

* Initial release
