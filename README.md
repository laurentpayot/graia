# 🌄 Graia

Neural network implemented in Futhark

## Prerequisites

To run Graia, you will need:

- [Futhark](https://futhark.readthedocs.io/en/stable/installation.html#installing-from-a-precompiled-snapshot)

On Debian/Ubuntu systems:

- OpenCL
  - If no OpenCL device is listed with `clinfo -l`, install `pocl-opencl-icd`
  - If missing *CL/cl.h* error, install `opencl-headers`
  - If missing *-lOpenCL* error, create an OpenCL link: `sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so`
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/)
