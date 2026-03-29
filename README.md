[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](license.md)
[![Build Status](https://github.com/lumicks/py-optics/actions/workflows/integrity_check.yml/badge.svg?branch=main)](https://github.com/lumicks/py-optics/actions/workflows/integrity_check.yml?query=branch%3Amain)

![header](https://raw.githubusercontent.com/lumicks/py-optics/main/docs/header.png)

# py-optics

Python code to aid with calculations for nanophotonics. Historically, Py-optics revolves around focused laser beams, with and without a spherical particle. The main modules are (full path `lumicks.pyoptics.<module_name>`):

- `psf`: code that calculates the point spread function (electromagnetic field distribution) of a focused laser beam with arbitrary wavefront. The code takes polarization into account and is suitable for high-NA cases.
- `trapping`: code that calculates 
  - The electromagnetic field distribution inside and outside of a spherical particle at an arbitrary distance relative to a focused laser. The laser beam that generates the focus can have an arbitrary wavefront. 
  - Forces on the particle.
- `farfield_transform`: code to transform the near field to the far field.
- `field_distributions`: electric and magnetic dipole field distributions for dipoles with arbitrary orientation.

Where possible, a multi-threaded implementation, transparently compiled with [Numba](https://numba.pydata.org), is used to speed up numerically-intense parts of a calculation. 

All code is developed in-house at LUMICKS. This is alpha-level software: care is taken to verify calculated results against analytical models and/or reference implementations, but no warranties of any kind are given (see also the [license](LICENSE.md)).

Py-optics is tested to be compatible with Python version 3.10 to 3.14. However, at the time of writing, free-threading builds are **not** supported. Py-optics runs on Linux, Windows and MacOS, and is free to use under the conditions of the [Apache-2.0 open source license](LICENSE.md).


## How to get started

1. Create and activate a virtual environment. With conda:

       $ conda create -n py-optics python=3.13
       $ conda activate py-optics
   
2. Clone the repository:

       $ git clone https://github.com/lumicks/py-optics
       $ cd py-optics

3. Install the *pyoptics* package:

       $ pip install -e .

   Or, if you want to install the additional dependencies for the example Notebooks:

       $ pip install -e .[examples]
   
   To run the benchmarks, you'll need to install extra packages for testing:
   
	   $ pip install -e . --group dev

Example Jupyter Notebooks are available in the `examples` directory:

4. Run the Notebook:

       $ cd examples
       $ jupyter notebook
