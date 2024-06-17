# Changelog

## v0.5.0 | TBD

### Changes
* Bumped minimum version of support Python to 3.9, to easily accomodate typing
* Bumped maximum tested version of Python to 3.12

### Improvements

* Added functionality to shift the coordinate system for `fast_psf()`
* Added Pytest tests for example notebooks
* Added Pytest test for czt equivalence between FFT and czt for specific cases

### Bug fixes
* Fixed broken czt transform for array inputs
* Fixed bug where the electric field was not correct as a function of z in `psf.reference.focused_dipole_ref`
* Fixed bug where the Z grid points were not returned even with `return_grid` being True in `psf.reference.focused_dipole_ref`

## v0.4.0 | 2023-08-19

* Initial release
