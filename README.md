# stensor − cross-platform GPU tensor library with Rust and Slang

The goal of **stensor** (pronounced s-tensor, aka, Slang tensor) is to essentially be "[**nalgebra**](https://nalgebra.rs) on the
gpu". It aims (but it isn’t there yet) to expose linear algebra operations (including BLAS-like and LAPACK-like
operations) as well as geometric types (quaternions, similarities, etc.) as Slang shaders and kernels.

> **Warning**
**stensor** is still very incomplete and under heavy development and is lacking many features.

See also the README of [slang-hal](https://github.com/dimforge/slang-hal/blob/main/README.md) for information on
supported platforms.

### Using Slang

In order to compile and run any slang project, be sure to define the `SLANG_DIR` environment variable:
1. Download the Slang compiler libraries for your platform: https://github.com/shader-slang/slang/releases/tag/v2025.16
2. Unzip the downloaded directory, and use its path as value to the `SLANG_DIR` environment variable: `SLANG_DIR=/path/to/slang`.
   Note that the variable must point to the root of the slang installation (i.e. the directory that contains `bin` and `lib`).
   We recommend adding that as a system-wide environment variables so that it also becomes available to your IDE.