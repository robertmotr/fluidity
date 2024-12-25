#include "physics.h"

namespace GPU {

    __global__ void bilinearInterpolationGpu(Particle *particles, uint32_t x, uint32_t y);

    __global__ void computeDivergenceGpu(Particle *particles, uint32_t widthSz,
                                        uint32_t heightSz);

    __global__ void computeAdvectionGpu(Particle *particles, uint32_t widthSz, 
                                        uint32_t heightSz);

    __global__ void computeDiffusionGpu(Particle *particles, uint32_t widthSz, 
                                        uint32_t heightSz, float diffusionRate, uint32_t iterations);

    __global__ void computePressureProjectionGpu(Particle *particles, uint32_t widthSz, 
                                                uint32_t heightSz, uint32_t iterations);

    __global__ void handleCollisionsGpu(Particle *particles, uint32_t widthSz, 
                                        uint32_t heightSz, bool freeSlip);
}