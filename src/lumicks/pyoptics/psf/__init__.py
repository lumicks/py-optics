import warnings

with warnings.catch_warnings():  # TODO: Remove deprecated functions and warnings
    warnings.simplefilter("ignore")
    from .fast import fast_gauss, fast_psf

__all__ = ["fast_gauss", "fast_psf"]
