#pragma once

#include "pch.h"

extern float dt;
extern bool useGpu;
extern bool freeSlip;
extern uint32 cellSize;

constexpr Vec2<float> gravity = Vec2<float>(0.0f, -9.8f);

template<typename T>
class Vec2 {
private:

    T values[2] = {0};

    static_assert(is_same(T, float) || is_same(T, uint32) || 
                  is_same(T, int) || is_same(T, float),
                  "Type used inside vector must be of unsigned int, int, or float.");    

    // for optimal memory accesses on the GPU
    static_assert(128 % sizeof(Vector) == 0, "Vector struct size needs to divide evenly into 128 bytes.");
public:
    Vec2(T x, T y) {
        values[0] = x;
        values[1] = y;
    }

    Vec2(T val) {
        values[0] = val;
        values[1] = val;
    }

    Vec2() {}

    // operator overloads start here
    T& operator[](size_t index) {
        if(index > 1) {
            printf("ERROR: Vector index out of bounds in setter, got: %d\n", index);
            return 0;
        }
        return values[index];
    }

    const T& operator[](size_t index) const {
        if(index > 1) {
            printf("ERROR: Vector index out of bounds in accessor, got: %d\n", index);
        }
        return values[index];
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec2& vec) {
        os << "[";
        os << values[0] << ", ";
        os << values[1] << "]";
        return os;
    }

    // to support vec + scalar 
    friend Vec2 operator+(const Vec2& vec, T scalar) {
        return Vec2(vec[0] + scalar, vec[1] + scalar);
    }

    // to support scalar + vec
    friend Vec2 operator+(T scalar, const Vec2& vec) {
        return Vec2(vec[0] + scalar, vec[1] + scalar);
    }

    // to support vec - scalar
    friend Vec2 operator-(const Vec2& vec, T scalar) {
        return Vec2(vec[0] - scalar, vec[1] - scalar);
    }

    // to support scalar - vec
    friend Vec2 operator-(T scalar, const Vec2& vec) {
        return Vec2(scalar - vec[0], scalar - vec[1]);
    }

    // to support scalar * vec
    friend Vec2 operator*(T scalar, const Vec2& vec) {
        return Vec2(vec[0] * scalar, vec[1] * scalar);
    }

    // to support vec * scalar
    friend Vec2 operator*(const Vec2& vec, T scalar) {
        return Vec2(vec[0] * scalar, vec[1] * scalar);
    }

    friend T dotProduct(const Vec2& a, const Vec2& b) {
        return a[0] * b[0] + a[1] * b[1];
    }

    friend T magnitude(const Vec2& a) {
        return sqrt(pow(values[0], 2) + pow(values[1], 2));
    } 
}

class Particle {
public:
    Vec2<uint32> position(0, 0);
    Vec2<float> velocity(0.0f, 0.0f);
    float density = 0.0f;
    float pressure = 0.0f;
    unsigned char rgb[4] = {0, 0, 0, 0};
    // no need for A value, but we pad rgb to 4 bytes to get
    // an even number for struct size, in order to ensure a list of particles
    // can fit in a cache line

    Particle() {} // already initialized no need to do anything 

private:
    // for optimal memory accesses on the GPU
    static_assert(128 % sizeof(Particle) == 0, "Particle struct size needs to divide evenly into 128 bytes.");
} 

__device__ __host__ clamp(Vec2 &vec, uint32 clamp, uint32 boundary);

__host__ void computeSimulationTick(Particle *particles, uint32 widthSz, 
                                    uint32 heightSz);

namespace GPU {

    __global__ void bilinearInterpolationGpu(Particle *particles, uint32 x, uint32 y);

    __global__ void computeDivergenceGpu(Particle *particles, uint32 widthSz,
                                        uint32 heightSz);

    __global__ void computeAdvectionGpu(Particle *particles, uint32 widthSz, 
                                        uint32 heightSz);

    __global__ void computeDiffusionGpu(Particle *particles, uint32 widthSz, 
                                        uint32 heightSz, float diffusionRate, uint32 iterations);

    __global__ void computePressureProjectionGpu(Particle *particles, uint32 widthSz, 
                                                uint32 heightSz, uint32 iterations);

    __global__ void handleCollisionsGpu(Particle *particles, uint32 widthSz, 
                                        uint32 heightSz, bool freeSlip);
}

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

