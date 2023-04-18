# py-optics

Python code to aid with calculations for nanophotonics. Currently it contains:

- *fast_psf_calc*: code that calculates the point spread function (electromagnetic field distribution) of a focused laser beam with arbitrary wavefront. The code takes polarization into account
- *mie_calc*: code that calculates the electromagnetic field distribution of a spherical particle at an arbitrary distance relative to a focused laser

All code is developed in-house at LUMICKS.


## How to get started

1. Create and activate a virtual environment. With conda:

       $ conda create -n py-optics python=3.8
       $ conda activate py-optics

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
