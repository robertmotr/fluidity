#include "physics.h"

namespace CPU {

    __host__ void bilinearInterpolationCpu(Particle *particles, uint32 x, uint32 y);

    __host__ void computeDivergenceCpu(Particle *particles, uint32 widthSz,
                                    uint32 heightSz);

    __host__ void computeAdvectionCpu(Particle *particles, uint32 widthSz,
                                    uint32 heightSz);

    __host__ void computeDiffusionCpu(Particle *particles, uint32 widthSz, 
                                    uint32 heightSz, float diffusionRate, uint32 iterations);

    __host__ void computePressureProjectionCpu(Particle *particles, uint32 widthSz, 
                                            uint32 heightSz, uint32 iterations);

    __host__ void handleCollisionsCpu(Particle *particles, uint32 widthSz, 
                                    uint32 heightSz, bool freeSlip);

}