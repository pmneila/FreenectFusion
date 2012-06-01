
#include "MarchingCubes.h"

#include <stdio.h>
#include <string.h>
#include "cudautils.h"
#include "cudamath.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#include "tables.h"

// textures containing look-up tables
texture<uint, 1, cudaReadModeElementType> edgeTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;

// volume data
texture<float, 1, cudaReadModeElementType> volumeTex;

// sample volume data set at a point
__device__
float sampleVolume(uint3 p, uint3 gridSize)
{
    p.x = min(p.x, gridSize.x - 1);
    p.y = min(p.y, gridSize.y - 1);
    p.z = min(p.z, gridSize.z - 1);
    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
//    return (float) data[i] / 255.0f;
    return tex1Dfetch(volumeTex, i);
}

// compute position in 3d grid from 1d index
// only works for power of 2 sizes
__device__
uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
{
    uint3 gridPos;
    gridPos.x = i & gridSizeMask.x;
    gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
    gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
    return gridPos;
}

// classify voxel based on number of vertices it will generate
// one thread per voxel
__global__ void
classifyVoxel(uint* voxelVerts, uint *voxelOccupied,
              uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
              float isoValue)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
    
    uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
    
    // read field values at neighbouring grid vertices
    float field[8];
    field[0] = sampleVolume(gridPos, gridSize);
    field[1] = sampleVolume(gridPos + make_uint3(1, 0, 0), gridSize);
    field[2] = sampleVolume(gridPos + make_uint3(1, 1, 0), gridSize);
    field[3] = sampleVolume(gridPos + make_uint3(0, 1, 0), gridSize);
    field[4] = sampleVolume(gridPos + make_uint3(0, 0, 1), gridSize);
    field[5] = sampleVolume(gridPos + make_uint3(1, 0, 1), gridSize);
    field[6] = sampleVolume(gridPos + make_uint3(1, 1, 1), gridSize);
    field[7] = sampleVolume(gridPos + make_uint3(0, 1, 1), gridSize);
    
    // calculate flag indicating if each vertex is inside or outside isosurface
    uint cubeindex;
	cubeindex =  uint(field[0] < isoValue); 
	cubeindex += uint(field[1] < isoValue)*2; 
	cubeindex += uint(field[2] < isoValue)*4; 
	cubeindex += uint(field[3] < isoValue)*8; 
	cubeindex += uint(field[4] < isoValue)*16; 
	cubeindex += uint(field[5] < isoValue)*32; 
	cubeindex += uint(field[6] < isoValue)*64; 
	cubeindex += uint(field[7] < isoValue)*128;
    
    // read number of vertices from texture
    uint numVerts = tex1Dfetch(numVertsTex, cubeindex);
    
    if (i < numVoxels) {
        voxelVerts[i] = numVerts;
        voxelOccupied[i] = (numVerts > 0);
    }
}

void MarchingCubesTextures::allocateTextures()
{
    cudaSafeCall(cudaMalloc((void**)&mEdgeTableGpu, 256*sizeof(uint)));
    cudaSafeCall(cudaMemcpy(mEdgeTableGpu, edgeTable, 256*sizeof(uint), cudaMemcpyHostToDevice));
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaSafeCall(cudaBindTexture(0, edgeTex, mEdgeTableGpu, channelDesc) );
    
    cudaSafeCall(cudaMalloc((void**)&mTriTableGpu, 256*16*sizeof(uint)));
    cudaSafeCall(cudaMemcpy(mTriTableGpu, triTable, 256*16*sizeof(uint), cudaMemcpyHostToDevice) );
    cudaSafeCall(cudaBindTexture(0, triTex, mTriTableGpu, channelDesc));

    cudaSafeCall(cudaMalloc((void**)&mNumVertsTableGpu, 256*sizeof(uint)));
    cudaSafeCall(cudaMemcpy(mNumVertsTableGpu, numVertsTable, 256*sizeof(uint), cudaMemcpyHostToDevice) );
    cudaSafeCall(cudaBindTexture(0, numVertsTex, mNumVertsTableGpu, channelDesc));
}

void MarchingCubes::bindVolumeTexture()
{
    cudaSafeCall(cudaBindTexture(0, volumeTex, mVolumeGpu, cudaCreateChannelDesc<float>()));
}

void MarchingCubes::classifyVoxels(float isovalue)
{
    dim3 threads(128);
    dim3 grid(mNumVoxels/threads.x, 1, 1);
    ::classifyVoxel<<<grid, threads>>>(mVoxelVertsGpu, mVoxelOccupiedGpu,
                                        mGridSize, mGridSizeShift, mGridSizeMask,
                                        mNumVoxels, isovalue);
    cudaSafeCall(cudaGetLastError());
}

void thrustScanWrapper(unsigned int* output, unsigned int* input, int numElements)
{
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input), 
                           thrust::device_ptr<unsigned int>(input + numElements),
                           thrust::device_ptr<unsigned int>(output));
}
