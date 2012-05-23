
#ifndef _CUDAUTILS_H
#define _CUDAUTILS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

inline void __cudaSafeCall(cudaError err, const char* file, int line)
{
    if(err != cudaSuccess)
    {
        std::stringstream ss;
        ss << "Error in file " << file << " line " << line << ": " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

#define cudaSafeCall(err) __cudaSafeCall((err), __FILE__, __LINE__)

#endif // _CUDAUTILS_H
