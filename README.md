# 🌄 Graia

Neural network implemented in Futhark and F#

## Prerequisites

To run Graia, you will need:

- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download)
- [Futhark](https://futhark.readthedocs.io/en/stable/installation.html#installing-from-a-precompiled-snapshot)

On Debian/Ubuntu systems:

- `gcc`
- OpenCL
  - If no OpenCL device is listed with `clinfo -l`, install `pocl-opencl-icd`
  - If missing *CL/cl.h* error, install `opencl-headers`
  - If missing *-lOpenCL* error, create an OpenCL link: `sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so`

## Usage

### Build

- `make`

## Next steps

- use [Basel](https://bazel.build/start/cpp) instead of Makefile?
