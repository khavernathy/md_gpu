A video demonstrating GPU MD using this code can be found here: https://www.youtube.com/watch?v=IPQzY34eCt4

This is a GPU MD code.
Douglas Franz and Alfredo Peguero
2016.
==================================

To download, use: 

`git clone https://github.com/khavernathy/md_gpu`

=========== CPU =======================

To compile (so far): 

`g++ FM3.1.cpp -lm -o t -I. -std=c++11`

To run: 

`./t`

========== GPU ========================

To compile (on a CUDA host):

`module load apps/cuda/7.5` // or whatever CUDA module you have/need

`deviceQuery` (optional, to get device information)

Access v_final folder for the complete program.

`nvcc md_gpu.cu -o executable` // to recompile. An already compiled executable "go" is included.

To run: 

`./executable [filename] [N particles, int] [box x, double] [box y, double] [box z, double] [block size, int]`

e.g.

`./go water_500.dat 1527 30 30 30 256` // The box dimensions are irrelevant (for now). Will be used later for PBC.
