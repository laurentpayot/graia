# 🌄 Graia

An *experimental* neural network library.

## Prerequisites

To build Graia on a Debian/Ubuntu system, you will need:

- [Futhark](https://futhark.readthedocs.io/en/stable/installation.html#installing-from-a-precompiled-snapshot)
- [Futhark FFI](https://github.com/pepijndevos/futhark-pycffi) `pip install futhark-ffi`
- OpenCL
  - If no OpenCL device is listed with `clinfo -l`, install `pocl-opencl-icd`
  - If missing *CL/cl.h* error, install `opencl-headers`
  - If missing *-lOpenCL* error, create an OpenCL link: `sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so`

Then simply run `make` to test and compile the Futhark files to the OpenCL library used by the `Graia` Python class.

## Jupyter Notebooks

To be sure to have all the Python packages needed, [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/) is highly recommended.

## TODO

- change the weights at the same time the outputs are computed
- find a way to use in-place updates to change the weights
- random weight initialization in GPU as CL array https://documen.tician.de/pyopencl/array.html#module-pyopencl.clrandom
- save/load model https://documen.tician.de/pyopencl/array.html#pyopencl.array.Array.get then https://numpy.org/devdocs/reference/generated/numpy.save.html
- tests
