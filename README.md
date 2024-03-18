# 🌄 Graia

An *experimental* neural network library.

## Prerequisites

To run Graia in a Jupyter Notebook on a Debian/Ubuntu system, you will need:

- [Futhark](https://futhark.readthedocs.io/en/stable/installation.html#installing-from-a-precompiled-snapshot)
- OpenCL
  - If no OpenCL device is listed with `clinfo -l`, install `pocl-opencl-icd`
  - If missing *CL/cl.h* error, install `opencl-headers`
  - If missing *-lOpenCL* error, create an OpenCL link: `sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so`
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/)

## TODO

- random weight initialization in GPU as CL array https://documen.tician.de/pyopencl/array.html#module-pyopencl.clrandom
- save/load model https://documen.tician.de/pyopencl/array.html#pyopencl.array.Array.get then https://numpy.org/devdocs/reference/generated/numpy.save.html
- tests
