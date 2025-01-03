#include "physics.h"
#include "globals.h"

namespace CPU {

    __host__ void bilinearInterpolation(Particle *particles, uint32_t x, uint32_t y) {}

    __host__ void computeDivergence(Particle *particles, uint32_t width,
                                    uint32_t height, std::vector<float> &divField) {

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

    __host__ void solvePressure(Particle *particles, std::vector<float> &divField, uint32_t width,
                                uint32_t height, uint32_t iterations) {
        
        float alpha = -1.0f / std::pow(PhysicsConfig::cellSize, 2); // Coefficients for Laplace operator
        float beta = -4.0f / std::pow(PhysicsConfig::cellSize, 2);
        
        for (uint32_t iter = 0; iter < iterations; ++iter) {
            for (uint32_t j = 1; j < height - 1; ++j) {
                for (uint32_t i = 1; i < width - 1; ++i) {
                    uint32_t idx = i + j * width;

                    particles[idx].pressure = (divField[idx] - alpha * (particles[idx + 1].pressure +
                                               particles[idx - 1].pressure + particles[idx + width].pressure + 
                                               particles[idx - width].pressure)) / beta;
                }
            }
        }
    }

    __host__ void computeAdvection(Particle *particles, uint32_t width,
                                   uint32_t height) {}

    __host__ void computeDiffusion(Particle *particles, uint32_t width, uint32_t height, 
                                   float diffusionRate, uint32_t iterations) {}

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

    __host__ void handleCollisions(Particle *particles, uint32_t width, 
                                   uint32_t height, bool freeSlip) {}

}

void computeSimulationTick(Particle *particles, uint32_t width, 
                           uint32_t height) {
    
    static std::vector<float> divField(width * height, 0.0f);

    if(RenderConfig::useGpu) {
        // call GPU functions
    } else {
        // call CPU functions
        CPU::computeDivergence(particles, width, height, divField);
        CPU::computePressureProjection(particles, width, height);
        CPU::computeAdvection(particles, width, height);
        CPU::computeDiffusion(particles, width, height, 
                              PhysicsConfig::diffusionRate, PhysicsConfig::iterations);
        CPU::handleCollisions(particles, width, height, PhysicsConfig::freeSlip);
    }
}