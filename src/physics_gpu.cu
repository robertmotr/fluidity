#include "physics.h"

namespace GPU {

    __global__ void bilinearInterpolation(Particle *particles, uint32_t x, uint32_t y);

    __global__ void computeDivergence(Particle *particles, uint32_t widthSz,
                                      uint32_t heightSz);

    __global__ void computeAdvection(Particle *particles, uint32_t widthSz, 
                                     uint32_t heightSz);

    __global__ void computeDiffusion(Particle *particles, uint32_t widthSz, 
                                     uint32_t heightSz, float diffusionRate, uint32_t iterations);

    __global__ void computePressureProjection(Particle *particles, uint32_t widthSz, 
                                              uint32_t heightSz, uint32_t iterations);

    __global__ void handleCollisions(Particle *particles, uint32_t widthSz, 
                                     uint32_t heightSz, bool freeSlip);
}