#include "physics.h"
#include "globals.h"

namespace CPU {

    __host__ void bilinearInterpolation(Particle *particles, uint32_t x, uint32_t y) {}

    __host__ void computeDivergence(Particle *particles, uint32_t widthSz,
                                    uint32_t heightSz) {}

    __host__ void computeAdvection(Particle *particles, uint32_t widthSz,
                                   uint32_t heightSz) {}

    __host__ void computeDiffusion(Particle *particles, uint32_t widthSz, 
                                   uint32_t heightSz, float diffusionRate, uint32_t iterations) {}

    __host__ void computePressureProjection(Particle *particles, uint32_t width, 
                                            uint32_t height) {

        for (uint32_t j = 1; j < height - 1; ++j) {
            for (uint32_t i = 1; i < width - 1; ++i) {
                uint32_t idx = i + j * width;
                particles[idx].velocity[0] = (particles[idx + 1].pressure - particles[idx].pressure) / PhysicsConfig::cellSize;
                particles[idx].velocity[1] = (particles[idx + width].pressure - particles[idx].pressure) / PhysicsConfig::cellSize;
            }
        }
    }

    __host__ void handleCollisions(Particle *particles, uint32_t widthSz, 
                                   uint32_t heightSz, bool freeSlip) {}

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