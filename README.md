[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](license.md)
[![Build Status](https://github.com/lumicks/py-optics/actions/workflows/integrity_check.yml/badge.svg?branch=main)](https://github.com/lumicks/py-optics/actions/workflows/integrity_check.yml?query=branch%3Amain)

# py-optics

Python code to aid with calculations for nanophotonics. Currently it contains:

- *lumicks.pyoptics.psf*: code that calculates the point spread function (electromagnetic field distribution) of a focused laser beam with arbitrary wavefront. The code takes polarization into account
- *lumicks.pyoptics.trapping*: code that calculates the electromagnetic field distribution of a spherical particle at an arbitrary distance relative to a focused laser

All code is developed in-house at LUMICKS.

Py-optics is free to use under the conditions of the [Apache-2.0 open source license](LICENSE.md).


## How to get started

1. Create and activate a virtual environment. With conda:

       $ conda create -n py-optics python=3.8
       $ conda activate py-optics
   
   On Windows, there is a potential performance improvement if numpy and scipy are installed with conda, as they are optimized to use the Intel MKL libraries.

2. Clone the repository:

       $ git clone https://github.com/lumicks/py-optics
       $ cd py-optics

3. Install the *pyoptics* package:

       $ pip install -e .

   Or, if you want to install the additional dependencies for the example Notebooks:

       $ pip install -e .[examples]
   
   To run the benchmarks, you'll need to install extra packages for testing:
   
	   $ pip install -e .[testing]

Example Jupyter Notebooks are available in the `examples` directory:

4. Run the Notebook:

       $ cd examples
       $ jupyter notebook
