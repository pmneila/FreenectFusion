
#include "MarchingCubes.h"

#include "FreenectFusion.h"

#include "cudautils.h"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>

#include <stdexcept>

MarchingCubesTextures* MarchingCubesTextures::instance = 0;

MarchingCubesTextures::MarchingCubesTextures()
{
    if(MarchingCubesTextures::instance != 0)
        throw std::runtime_error("An instance of MarchingCubesTextures already exists.");
    
    MarchingCubesTextures::instance = this;
    allocateTextures();
}

MarchingCubesTextures::~MarchingCubesTextures()
{
    cudaSafeCall(cudaFree(mEdgeTableGpu));
    cudaSafeCall(cudaFree(mTriTableGpu));
    cudaSafeCall(cudaFree(mNumVertsTableGpu));
}

MarchingCubes::MarchingCubes(VolumeFusion* volume)
    : mSideLog(volume->getSideLog()),
    mSide(volume->getSide()), mVolumeGpu(volume->getFGpu()),
    mGridSize(make_uint3(mSide, mSide, mSide)),
    mGridSizeMask(make_uint3(mGridSize.x-1, mGridSize.y-1, mGridSize.z-1)),
    mGridSizeShift(make_uint3(0, mSideLog, mSideLog+mSideLog)),
    mNumVoxels(mSide*mSide*mSide), mMaxVertices(mSide*mSide*100),
    mActiveVoxels(0), mTotalVertices(0)
{
    // Allocate vertex and normal buffers.
    glGenBuffers(1, &mVertexBuffer);
    glGenBuffers(1, &mNormalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, mMaxVertices*4*sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
    glBufferData(GL_ARRAY_BUFFER, mMaxVertices*4*sizeof(float), 0, GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(mVertexBuffer);
    cudaGLRegisterBufferObject(mNormalBuffer);
    
    // Allocate device memory.
    size_t memsize = sizeof(unsigned int)*mNumVoxels;
    cudaSafeCall(cudaMalloc((void**)&mVoxelVertsGpu, memsize));
    cudaSafeCall(cudaMalloc((void**)&mVoxelVertsScanGpu, memsize));
    cudaSafeCall(cudaMalloc((void**)&mVoxelOccupiedGpu, memsize));
}

MarchingCubes::~MarchingCubes()
{
    // Deallocate buffers.
    cudaSafeCall(cudaGLUnregisterBufferObject(mVertexBuffer));
    cudaSafeCall(cudaGLUnregisterBufferObject(mNormalBuffer));
    glDeleteBuffers(1, &mVertexBuffer);
    glDeleteBuffers(1, &mNormalBuffer);
    
    // Free device memory.
    cudaSafeCall(cudaFree(mVoxelVertsGpu));
    cudaSafeCall(cudaFree(mVoxelVertsScanGpu));
    cudaSafeCall(cudaFree(mVoxelOccupiedGpu));
}

void MarchingCubes::computeMC(float isovalue)
{
    bindVolumeTexture();
    classifyVoxels(isovalue);
    thrustScanWrapper(mVoxelVertsScanGpu, mVoxelVertsGpu, mNumVoxels);
    
    mTotalVertices = thrust::device_ptr<unsigned int>(mVoxelVertsScanGpu)[mNumVoxels-1]
                     + thrust::device_ptr<unsigned int>(mVoxelVertsGpu)[mNumVoxels-1];
    std::cout << mTotalVertices << std::endl;
}
