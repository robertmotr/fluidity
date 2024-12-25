#include "physics.h"

namespace GPU {

    __global__ void bilinearInterpolationGpu(Particle *particles, uint32 x, uint32 y);

    __global__ void computeDivergenceGpu(Particle *particles, uint32 widthSz,
                                        uint32 heightSz);

    __global__ void computeAdvectionGpu(Particle *particles, uint32 widthSz, 
                                        uint32 heightSz);

    __global__ void computeDiffusionGpu(Particle *particles, uint32 widthSz, 
                                        uint32 heightSz, float diffusionRate, uint32 iterations);

    __global__ void computePressureProjectionGpu(Particle *particles, uint32 widthSz, 
                                                uint32 heightSz, uint32 iterations);

    __global__ void handleCollisionsGpu(Particle *particles, uint32 widthSz, 
                                        uint32 heightSz, bool freeSlip);
}