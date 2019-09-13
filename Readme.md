RBMonGPU
========

This library implements the [algorithm of Carleo and Troyer](https://arxiv.org/abs/1606.02318) for respresenting a wavefunction as a restricted Boltzmann machine on the GPU. The code is written in python/C++ and makes heavily use of inline-functions. However it is *not* a header-only library. This means you don't have to compile using the CUDA-compiler for making use of this library within your own code.
