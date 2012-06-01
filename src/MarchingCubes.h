
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
    int mActiveVoxels, mActiveVertices;
    
    const float* mVolumeGpu;
    unsigned int* mVoxelVertsGpu;
    unsigned int* mVoxelVertsScanGpu;
    unsigned int* mVoxelOccupiedGpu;
    unsigned int* mVoxelOccupiedScanGpu;
    unsigned int* mCompVoxelArrayGpu;
    
    unsigned int mVertexBuffer, mNormalBuffer;
    
    void bindVolumeTexture();
    void classifyVoxels(float isovalue);
    void compactVoxels();
    void generateTriangles(float isovalue);
    
public:
    MarchingCubes(int sidelog);
    ~MarchingCubes();
    
    void computeMC(float isovalue=0.f);
    
    inline unsigned int getGLVertexBuffer() const {return mVertexBuffer;}
    inline unsigned int getGLNormalBuffer() const {return mNormalBuffer;}
    inline int getActiveVertices() const {return mActiveVertices;}
};

#endif // _MARCHINGCUBES_H
