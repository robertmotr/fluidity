#pragma once

#include "pch.h"

template<typename T>
class Vec2 {
private:

    T values[2] = {0};

    static_assert(std::is_same<T, float>::value || std::is_same<T, uint32_t>::value ||
                  std::is_same<T, int>::value, "Type used inside vector must be of unsigned int, int, or float.");    

public:

    Vec2(T x, T y) {
        values[0] = x;
        values[1] = y;
    }

    Vec2(T val) {
        values[0] = val;
        values[1] = val;
    }

    Vec2() = default;

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
        os << vec.values[0] << ", ";
        os << vec.values[1] << "]";
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

    // vec + vec
    Vec2 operator+(const Vec2& vec) {
        return Vec2(values[0] + vec[0], values[1] + vec[1]);
    }

    // to support vec - scalar
    friend Vec2 operator-(const Vec2& vec, T scalar) {
        return Vec2(vec[0] - scalar, vec[1] - scalar);
    }

    // to support scalar - vec
    friend Vec2 operator-(T scalar, const Vec2& vec) {
        return Vec2(scalar - vec[0], scalar - vec[1]);
    }

    // vec - vec
    Vec2 operator-(const Vec2& vec) {
        return Vec2(values[0] - vec[0], values[1] - vec[1]);
    }

    // to support scalar * vec
    friend Vec2 operator*(T scalar, const Vec2& vec) {
        return Vec2(vec[0] * scalar, vec[1] * scalar);
    }

    // to support vec * scalar
    friend Vec2 operator*(const Vec2& vec, T scalar) {
        return Vec2(vec[0] * scalar, vec[1] * scalar);
    }

    // to support vec * vec
    Vec2 operator*(const Vec2& vec) {
        return Vec2(vec[0] * values[0], vec[1] * values[1]);
    }

    friend T dotProduct(const Vec2& a, const Vec2& b) {
        return a[0] * b[0] + a[1] * b[1];
    }

    friend T magnitude(const Vec2& vec) {
        return std::sqrt(std::pow(vec.values[0], 2) + pow(vec.values[1], 2));
    } 
};

static_assert(128 % sizeof(Vec2<uint32_t>) == 0, "Vector struct size needs to divide evenly into 128 bytes.");
static_assert(128 % sizeof(Vec2<float>) == 0, "Vector struct size needs to divide evenly into 128 bytes.");
static_assert(128 % sizeof(Vec2<int>) == 0, "Vector struct size needs to divide evenly into 128 bytes.");

class Particle {
public:
    Vec2<float> position = Vec2<float>(0.0f, 0.0f);
    Vec2<float> velocity = Vec2<float>(0.0f, 0.0f);
    float friction = 0.0f;
    float density = 0.0f;
    float pressure = 0.0f;
    unsigned char rgb[4] = {0, 0, 0, 0};
    // no need for A value, but we pad rgb to 4 bytes to get
    // an even number for struct size, in order to ensure a list of particles
    // can fit in a cache line

    Particle() = default;
};

static_assert(128 % sizeof(Particle) == 0, "Particle struct size needs to be 128 bytes.");

__host__ void computeSimulationTick(Particle *particles, uint32_t width, 
                                    uint32_t height);

namespace GPU {

    __global__ void bilinearInterpolation(Particle *particles, float x, float y);

    __global__ void computeDivergence(Particle *particles, uint32_t width,
                                      uint32_t height);

    __global__ void computeAdvection(Particle *particles, uint32_t width, 
                                     uint32_t height);

    __global__ void computeDiffusion(Particle *particles, uint32_t width, 
                                     uint32_t height, float diffusionRate, uint32_t iterations);

    __global__ void computePressureProjection(Particle *particles, uint32_t width, 
                                              uint32_t height);

    __global__ void handleCollisions(Particle *particles, uint32_t width, 
                                     uint32_t height, bool freeSlip);
}

namespace CPU {

    __host__ void bilinearInterpolation(Particle *particles, uint32_t x, uint32_t y);

    __host__ void computeDivergence(Particle *particles, uint32_t width,
                                    uint32_t height, std::vector<float> &divField);

    __host__ void computeAdvection(Particle *particles, uint32_t width,
                                   uint32_t height);

    __host__ void computeDiffusion(Particle *particles, uint32_t width, 
                                   uint32_t height, float diffusionRate, uint32_t iterations);

    __host__ void computePressureProjection(Particle *particles, uint32_t width, 
                                            uint32_t height);
  
    __host__ void solvePressure(Particle *particles, std::vector<float> &divField, uint32_t width,
                                uint32_t height, uint32_t iterations);

    __host__ void handleCollisions(Particle *particles, uint32_t width, 
                                   uint32_t height, bool freeSlip);

}

