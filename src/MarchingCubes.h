
#ifndef _MARCHINGCUBES_H
#define _MARCHINGCUBES_H

#include <vector_types.h>

class MarchingCubesTextures
{
private:
    void allocateTextures();
    unsigned int* mEdgeTableGpu;
    unsigned int* mTriTableGpu;
    unsigned int* mNumVertsTableGpu;
    
    static MarchingCubesTextures* instance;
    
public:
    MarchingCubesTextures();
    ~MarchingCubesTextures();
    
    static MarchingCubesTextures* getInstance();
};

class VolumeFusion;

void thrustScanWrapper(unsigned int* output, unsigned int* input, int numElements);

class MarchingCubes
{
private:
    int mSideLog, mSide;
    int mNumVoxels, mMaxVertices;
    uint3 mGridSize, mGridSizeMask, mGridSizeShift;
    int mActiveVoxels, mTotalVertices;
    
    const float* mVolumeGpu;
    unsigned int* mVoxelVertsGpu;
    unsigned int* mVoxelVertsScanGpu;
    unsigned int* mVoxelOccupiedGpu;
    //unsigned int* mVoxelOccupiedScanGpu;
    //unsigned int* mCompVoxelArrayGpu;
    
    unsigned int mVertexBuffer, mNormalBuffer;
    
    void bindVolumeTexture();
    void classifyVoxels(float isovalue);
    
public:
    MarchingCubes(VolumeFusion* volume);
    ~MarchingCubes();
    
    void computeMC(float isovalue=0.f);
};

#endif // _MARCHINGCUBES_H
