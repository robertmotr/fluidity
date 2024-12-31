#include "physics.h"
#include "globals.h"

namespace CPU {

    __host__ void bilinearInterpolation(Particle *particles, uint32_t x, uint32_t y) {}

    __host__ void computeDivergence(Particle *particles, uint32_t width,
                                    uint32_t height, std::array &divField) {

        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                uint32_t idx = i + j * width;
                float div = 0.0f;

                // x-direction (partial x / delta x)
                if(j < width - 1) {
                    div += (particles[idx + 1].velocity[0] - particles[idx].velocity[0]) / PhysicsConfig::cellSize;
                } else{
                    div += (0 - particles[idx].velocity[0]) / PhysicsConfig::cellSize;
                }
                
                // y-direction (partial y / delta y)
                if(i < height - 1) {
                    div += (particles[idx + width].velocity[1] - particles[idx].velocity[1]) / PhysicsConfig::cellSize;
                } else {
                    div += (0 - particles[idx].velocity[1]) / PhysicsConfig::cellSize;
                }

                divField[idx] = div;

            }
        }
    }
        

    __host__ void computeAdvection(Particle *particles, uint32_t widthSz,
                                   uint32_t heightSz) {}

    __host__ void computeDiffusion(Particle *particles, uint32_t widthSz, 
                                   uint32_t heightSz, float diffusionRate, uint32_t iterations) {}

    __host__ void computePressureProjection(Particle *particles, uint32_t widthSz, 
                                            uint32_t heightSz, uint32_t iterations) {}

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