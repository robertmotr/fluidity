#include "physics.h"
#include "globals.h"

namespace PhysicsConfig {
    float dt = 0.1f;
    bool freeSlip = false;
    uint32_t cellSize = 1;
    Vec2<float> customForce(0.0f, 0.0f);
    uint32_t simulationWidthCells = 64;
    uint32_t simulationHeightCells = 64;
}

namespace CPU {

    __host__ void bilinearInterpolationCpu(Particle *particles, uint32_t x, uint32_t y);

    __host__ void computeDivergenceCpu(Particle *particles, uint32_t widthSz,
                                    uint32_t heightSz);

    __host__ void computeAdvectionCpu(Particle *particles, uint32_t widthSz,
                                    uint32_t heightSz);

    __host__ void computeDiffusionCpu(Particle *particles, uint32_t widthSz, 
                                    uint32_t heightSz, float diffusionRate, uint32_t iterations);

    __host__ void computePressureProjectionCpu(Particle *particles, uint32_t widthSz, 
                                            uint32_t heightSz, uint32_t iterations);

    __host__ void handleCollisionsCpu(Particle *particles, uint32_t widthSz, 
                                    uint32_t heightSz, bool freeSlip);

}

void computeSimulationTick(Particle *particles, uint32_t widthSz, 
                           uint32_t heightSz) {
    if(RenderConfig::useGpu) {
        // call GPU functions
    } else {
        // call CPU functions
        CPU::computeDivergenceCpu(particles, widthSz, heightSz);
        CPU::computePressureProjectionCpu(particles, widthSz, heightSz, PhysicsConfig::iterations);
        CPU::computeAdvectionCpu(particles, widthSz, heightSz);
        CPU::computeDiffusionCpu(particles, widthSz, heightSz, 
                                 PhysicsConfig::diffusionRate, PhysicsConfig::iterations);
        CPU::handleCollisionsCpu(particles, widthSz, heightSz, PhysicsConfig::freeSlip);
    }
}