#pragma once

#include "physics.h"

namespace PhysicsConfig {
    extern float dt;
    extern bool freeSlip;
    extern float diffusionRate;
    extern uint32_t iterations;
    
    extern uint32_t cellSize;
    extern Vec2<float> customForce;

    extern uint32_t simulationWidthCells; 
    extern uint32_t simulationHeightCells; 

    extern const Vec2<float> gravity;
} 

namespace RenderConfig {
    constexpr uint32_t windowWidth = 1920;
    constexpr uint32_t windowHeight = 1080;
    const char* windowTitle = "Fluid Simulation";
    
    extern bool useGpu;
    extern uint32_t kernelGridSize;
    extern uint32_t kernelBlockSize;
}