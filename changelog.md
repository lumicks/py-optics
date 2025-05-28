# Changelog

## v0.7.0 | TBD

### Changes

* Bumped minimum version of Python to 3.10
* Bumped maximum tested version of Python to 3.13
* Removed chirped z-transform code in favor of Scipy's `CZT` class.
### Improvements

* Added Gauss-Legendre and Clenshaw-Curtis integration methods for force calculations
* Changed the default integration order to twice the number of Mie modes for the Lebedev-Laikov integration scheme when no integration order is given.
* Added the electromagnetic field distribution for magnetic dipoles at arbitrary orientations
* Added a method to determine the minimal sampling order of the back focal plane to `objective.Objective`

# Changes
* Renamed `integration_orders` to `integration_order`

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
