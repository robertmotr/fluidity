#include "globals.h"

namespace PhysicsConfig {
    float dt = 0.1f;
    bool freeSlip = false;
    uint32_t cellSize = 1;
    Vec2<float> customForce(0.0f, 0.0f);
    uint32_t simulationWidthCells = 64;
    uint32_t simulationHeightCells = 64;

    const Vec2<float> gravity(0.0f, -9.8f);
}

namespace RenderConfig {
    bool useGpu = false;
    uint32_t kernelGridSize = 1;
    uint32_t kernelBlockSize = 1;
}