#include <cudart.h>
#include <cstddef>

typedef unsigned int uint32;

float dt;
bool useGpu;
bool freeSlip;
uint32 cellSize;

struct Particle {
    Position pos = 0;
    float vx, vy = 0.0f;
    float density = 0.0f;
    float pressure = 0.0f;
    unsigned char rgb[4] = {0, 0, 0, 0};
    // no need for A value, but we pad rgb to 4 bytes to get
    // an even number for struct size, in order to ensure a list of particles
    // can fit in a cache line

    Particle() : position(0), vx(0.0f), vy(0.0f), density(0.0f),
                 pressure(0.0f), rgb(0) {}

    // for optimal memory accesses when it comes to cache lines
    static_assert(128 % sizeof(Particle) == 0, "Particle struct size needs to divide evenly into 128 bytes.");

} 

struct Position {
    uint32 x, y = 0;

    Position(uint32 x, uint32 y) : x(x), y(y) {}

    Position() : x(0), y(0) {}

    // for optimal memory accesses when it comes to cache lines
    static_assert(128 % sizeof(Position) == 0, "Position struct size needs to divide evenly into 128 bytes.");
}

__device__ __host__ clamp(Position &pos, uint32 clamp, uint32 boundary);

__host__ void bilinearInterpolationCpu();

__global__ void bilinearInterpolationGpu();

__host__ void divergenceCpu();

__global__ void divergenceGpu();

__host__ void advectionCpu(Particle *particles, uint32 widthSz,
                           uint32 heightSz);

__global__ void advectionGpu(Particle *particles, uint32 widthSz, 
                             uint32 heightSz);

__host__ void diffusionCpu(Particle *particles, uint32 widthSz, 
                           uint32 heightSz, float diffusionRate, uint32 iterations);

__global__ void diffusionGpu(Particle *particles, uint32 widthSz, 
                             uint32 heightSz, float diffusionRate, uint32 iterations);

__host__ void pressureProjectionCpu(Particle *particles, uint32 widthSz, 
                                    uint32 heightSz, uint32 iterations);

__global__ void pressureProjectionGpu(Particle *particles, uint32 widthSz, 
                                    uint32 heightSz, uint32 iterations);

__host__ void handleCollisionsCpu(Particle *particles, uint32 widthSz, 
                                 uint32 heightSz, bool freeSlip);

__global__ void handleCollisionsGpu(Particle *particles, uint32 widthSz, 
                                   uint32 heightSz, bool freeSlip);

__host__ void simulate(Particle *particles, uint32 size);


