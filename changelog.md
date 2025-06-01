# Changelog

## v0.7.0 | TBD

### Changes

* Bumped minimum version of Python to 3.10
* Bumped maximum tested version of Python to 3.13
* Removed Lebedev-Laikov code in favor of Scipy's `lebedev_rule` function.
* Removed chirped z-transform code in favor of Scipy's `CZT` class.
* Renamed `integration_orders` to `integration_order`
* Renamed `bfp_sampling_n` to `integration_order` for `psf.direct.focus_gauss` and `psf.direct.direct_psf`.

### Improvements

* Added Pytest for CZT-based far-field transform
* Added Gauss-Legendre and Clenshaw-Curtis integration methods for force calculations (in case orders higher than those supported by Lebedev-Laikov are required)
* Changed the default integration order to twice the number of Mie modes for the Lebedev-Laikov integration scheme when no integration order is given.
* Added the electromagnetic field distribution for magnetic dipoles at arbitrary orientations
* Added a method to determine the minimal sampling order of the back focal plane to `objective.Objective`
* Added a Gauss-Legendre-style integration scheme to `mathutils.integration` for integration over a disk or annulus. The application is for the back focal plane, and results in faster convergence compared to equidistant sampling, when using non-czt-based calculation methods to obtain a focus.

### Deprecations
* Deprecated the functions `psf.fast_psf` and `psf.fast_gauss` in favor of `psf.focus_czt` and `psf.focus_gaussian_czt`. The new functions are nearly the same as the deprecated versions, but take an `Objective` class instance instead of four parameters for `n_bfp`, `n_medium`, `NA` and `focal_length`. These parameters all describe properties of the objective used for focusing. The new name gives a clear indication of the computational method, the chirped-z transform (czt).

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
