# Changelog

## v0.7.0 | TBD

### Improvements

* Added integration schemes to `mathutils.integration` for integrating over a disk or annulus. The schemes `"peirce"` and `"lether"` are based on a Gauss-Legendre-style integration scheme, whereas the scheme `"takaki"` uses optimal placement of integration locations.
    1. William H. Peirce, "Numerical Integration Over the Planar Annulus,",  Journal of the Society for Industrial and Applied Mathematics, Vol. 5, No. 2 (Jun., 1957), pp. 66-73
    2. Frank G. Lether, "A Generalized Product Rule For The Circle,"  SIAM Journal on Numerical Analysis, Volume 8, Issue 2, Pages 249 - 253
    3. Nick Takaki, G. W. Forbes, and Jannick P. Rolland, "Schemes for cubature over the unit disk found via numerical optimization,", Journal of Computational and Applied Mathematics
* Added the ability to use new cylindrical integration schemes with the `trapping` module.
* Added Gauss-Legendre and Clenshaw-Curtis spherical integration methods for force calculations (in case orders higher than those supported by Lebedev-Laikov are required)
    1. Jörg Waldvogel, "Fast Construction of the Fejér and Clenshaw–Curtis Quadrature Rules," BIT Numer. Math. 46, 195–202 (2006)
    2. Kendall Atkinson, "Numerical integration on the sphere," The Journal of the Australian Mathematical Society Series B Applied Mathematics. 1982;23(3):332-347
* Added a method to automatically determine the minimal integration order to `objective.Objective`
    * Supported methods are `"peirce"` (new) and `"equidistant"` (only option until v0.7.0)
* Changed the default spherical integration order to twice the number of Mie modes for the Lebedev-Laikov integration scheme when no integration order is given. That is, `spherical_integration_order = Bead.number_of_modes * 2`
* Added the electromagnetic field distribution for magnetic dipoles at arbitrary orientations.
* Added `psf.czt` and `psf.quad`, two modules that contain functions to calculate the Point Spread Function (PSF) of arbitrary fields on the back focal plane of an objective. They use a new callback signature and and re-use what is already calculated by the `Objective` class. The names of the modules make the method of calculation more explicit, namely either the Chirped-Z Tranform (CZT) is used, or a 2D quadrature method is used. The `psf.quad` module supports a new integration method called "peirce" that is dedicated to integration over a circular domain, such as the back focal plane. It typically converges faster than the equidistant ("czt") methods, but the actual CZT implementations could still beat quadrature on raw speed.
* Added Pytest for CZT-based far-field transform

### Changes

* Bumped minimum version of Python to 3.10
* Bumped maximum tested version of Python to 3.13
* Removed Lebedev-Laikov code in favor of Scipy's `lebedev_rule` function.
* Removed chirped z-transform code in favor of Scipy's `CZT` class.
* Updated the signature of the callback used to calculate the field on the back focal plane (see examples for new type).
* Made the distinction between the number of spherical modes, the order of integration over the back focal plane and the order of integration over the sphere more explicit by renaming the parameters of the functions in the `trapping` module.
* Added the propery `Bead.number_of_modes`

### Deprecations

* Deprecated the modules `psf.fast` and `psf.direct`, and the functions `psf.fast_psf` and `psf.fast_gauss`. The new functions (see above) are nearly the same as the deprecated versions, but take an `Objective` class instance instead of four parameters for `n_bfp`, `n_medium`, `NA` and `focal_length`, and use the new callback signature. 
* Deprecated the property `Bead.number_of_orders` in favor of `Bead.number_of_modes`, to help distinguishing between integration order and spherical modes.


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
