#include "physics.h"
#include "globals.h"

namespace CPU {
    
    __host__ void addInput(Grid *grid, std::vector<Vec2<float>> &velocity, std::vector<float> &density) {}

    __host__ void bilinearInterpolation(Grid *grid, uint32_t x, uint32_t y) {
        int i = static_cast<int>(x);
        int j = static_cast<int>(y);

        i = std::clamp(i, 0, static_cast<int>(grid.width) - 2);
        j = std::clamp(j, 0, static_cast<int>(grid.height) - 2);

        float dx = x - i;
        float dy = y - j;

        uint32_t idx00 = grid.index(i, j);       
        uint32_t idx10 = grid.index(i + 1, j);   
        uint32_t idx01 = grid.index(i, j + 1);   
        uint32_t idx11 = grid.index(i + 1, j + 1); 

        // interpolate values
        float value = (1 - dx) * (1 - dy) * field[idx00] +
                    dx * (1 - dy) * field[idx10] +
                    (1 - dx) * dy * field[idx01] +
                    dx * dy * field[idx11];

        return value;
    }

    __host__ void computeDivergence(Grid *grid) {

        std::vector<Vec2<float>> &velocity = grid->velocity;

        for(int i = 0; i < grid->height; i++){
            for(int j = 0; j < grid->width; j++){
                uint32_t idx = i + j * grid->width;
                float div = 0.0f;

                // x-direction (partial x / delta x)
                if(j < grid->width - 1) {
                    div += (velocity[idx + 1][0] - velocity[idx][0]) / PhysicsConfig::cellSize;
                } else{
                    div += (0 - velocity[idx][0]) / PhysicsConfig::cellSize;
                }
                
                // y-direction (partial y / delta y)
                if(i < grid->height - 1) {
                    div += (velocity[idx + grid->width][1] - velocity[idx][1]) / PhysicsConfig::cellSize;
                } else {
                    div += (0 - velocity[idx][1]) / PhysicsConfig::cellSize;
                }

                grid->divField[idx] = div;

            }
        }
    }

    __host__ void solvePressure(Grid *grid, uint32_t iterations) {
        
        float alpha = -1.0f / std::pow(PhysicsConfig::cellSize, 2); // Coefficients for Laplace operator
        float beta = -4.0f / std::pow(PhysicsConfig::cellSize, 2);
        
        std::vector<float> &divField = grid->divField;
        std::vector<float> &pressure = grid->pressure;

        for (uint32_t iter = 0; iter < iterations; ++iter) {
            for (uint32_t j = 1; j < grid->height - 1; ++j) {
                for (uint32_t i = 1; i < grid->width - 1; ++i) {
                    uint32_t idx = i + j * grid->width;

                    if (i == 0 || j == 0 || i == grid->width - 1 || j == grid->height - 1) {
                        pressure[idx] = 0.0f; // Dirichlet boundary condition
                    }
                    else {
                        pressure[idx] = (divField[idx] - alpha * (pressure[idx + 1] +
                                     pressure[idx - 1] + pressure[idx + width] + 
                                     pressure[idx - width])) / beta;
                    }
                }
            }
        }
    }

    __host__ void computeAdvection(Grid *grid) {}

    __host__ void computeDiffusion(Grid *grid, float diffusionRate, uint32_t iterations) {}

    __host__ void computePressureProjection(Grid *grid) {

        std::vector<Vec2<float>> &velocity = grid->velocity;
        std::vector<float> &pressure = grid->pressure;

        for (uint32_t j = 1; j < grid->height - 1; ++j) {
            for (uint32_t i = 1; i < grid->width - 1; ++i) {
                uint32_t idx = i + j * grid->width;
                velocity[idx][0] = (pressure[idx + 1] - pressure[idx]) / PhysicsConfig::cellSize;
                velocity[idx][1] = (pressure[idx + grid->width] - pressure[idx]) / PhysicsConfig::cellSize;
            }
        }
    }

    __host__ void handleCollisions(Grid *grid, bool freeSlip) {}

}

void computeSimulationTick(Grid *grid) {
    
    if(RenderConfig::useGpu) {
        // call GPU functions
    } else {
        // call CPU functions
    }
}