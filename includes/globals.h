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

    constexpr Vec2<float> gravity = Vec2<float>(0.0f, -9.8f);
} 

namespace RenderConfig {
    constexpr uint32_t windowWidth = 800;
    constexpr uint32_t windowHeight = 800;
    const char* windowTitle = "Fluid Simulation";
    
    extern bool useGpu;
    extern uint32_t kernelGridSize;
    extern uint32_t kernelBlockSize;
}