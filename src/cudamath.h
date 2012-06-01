
#ifndef _CUDAMATH_H
#define _CUDAMATH_H

#include "cudautils.h"

inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}

inline __host__ __device__ float3 make_float3(float w)
{
    return make_float3(w, w, w);
}

inline __host__ __device__ float3 make_float3(uint3 a)
{
    return make_float3(a.x, a.y, a.z);
}

inline __device__ uint3 operator+(const uint3& a, const uint3& b)
{
    return make_uint3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ float length2(float2 v)
{
    return sqrtf(v.x*v.x + v.y*v.y);
}

inline __device__ float dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

inline __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}

__device__ float3 transform3(const float* matrix, float3 v);
__device__ float3 transform3_affine(const float* matrix, float3 v);
__device__ float3 transform3_affine_inverse(const float* matrix, float3 v);

inline __device__ float length(const float3& v)
{
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline __device__ float3 normalize(float3 v)
{
    float invLen = 1.0f / length(v);
    return invLen * v;
}

#endif // _CUDAMATH_H
