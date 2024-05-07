# 🌄 Graia

An *experimental* neural network library.

- *Not* using a retropropagation algorithm for training. Retropropagation works pretty well but the main goal of this project is to find a training algorithm that would work using only the information available *locally* to the nodes. Current status: not working 😅
- Written in [Futhark](https://futhark-lang.org/) to leverage OpenCL for GPU acceleration.
- Python API similar to [TensorFlow](https://www.tensorflow.org/).

## Prerequisites

To build Graia on a Debian/Ubuntu system, you need:

- [Futhark](https://futhark.readthedocs.io/en/stable/installation.html#installing-from-a-precompiled-snapshot)
- [Futhark FFI](https://github.com/pepijndevos/futhark-pycffi) `pip install futhark-ffi`
- OpenCL
  - Native GPU drivers are prefered but if no OpenCL device is listed with `clinfo -l` you can install `pocl-opencl-icd` (slower and sometimes buggy).
  - If missing *CL/cl.h* error, install `opencl-headers`.
  - If missing *-lOpenCL* error, create an OpenCL link: `sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so`.

Then simply run `make` to test and compile the Futhark files to the OpenCL library used by the `Graia` Python class.

## Jupyter Notebooks

To be sure to have all the Python packages needed, [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/) is highly recommended.
- Install [Futhark FFI](https://github.com/pepijndevos/futhark-pycffi): `pip install futhark-ffi`.
- As mentioned above, Native GPU drivers are prefered but if they don’t work you can Install [PoCL](http://portablecl.org/):`conda install conda-forge::pocl`.

## TODO

- Find a working training algorithm 😅
- Save/load model https://documen.tician.de/pyopencl/array.html#pyopencl.array.Array.get then https://numpy.org/devdocs/reference/generated/numpy.save.html
- More Futhark tests
