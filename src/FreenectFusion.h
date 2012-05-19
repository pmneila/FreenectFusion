
#ifndef _FREENECTFUSION_H
#define _FREENECTFUSION_H

#include <stdint.h>

class float2;

class MatrixGpu
{
private:
    int mSide;
    int mSize;
    float* mK;
    float* mKGpu;
    float* mKinv;
    float* mKinvGpu;
    
public:
    static MatrixGpu* newIntrinsicMatrix(const double* K);
    static MatrixGpu* newTransformMatrix(const double* T);
    
    MatrixGpu(int side, const double* K, const double* Kinv=0);
    ~MatrixGpu();
    
    inline operator const float*() const {return mK;}
    
    inline const float* get() const {return mK;}
    inline const float* getInverse() const {return mKinv;}
    inline const float* getGpu() const {return mKGpu;}
    inline const float* getInverseGpu() const {return mKinvGpu;}
};

class Measurement
{
private:
    int mWidth, mHeight, mNumVertices;
    float* mDepthGpu;
    uint16_t* mRawDepthGpu;
    mutable float* mDepth;
    
    MatrixGpu* mKdepth;
    
    unsigned int mVertexBuffer;
    unsigned int mNormalBuffer;
    int* mMaskGpu;
    
public:
    Measurement(int width, int height, const double* Kdepth);
    ~Measurement();
    
    void setDepth(uint16_t* depth);
    
    inline float* getDepthGpu() {return mDepthGpu;}
    inline const float* getDepthGpu() const {return mDepthGpu;}
    const float* getDepthHost() const;
    
    inline unsigned int getGLVertexBuffer() const {return mVertexBuffer;}
    inline unsigned int getGLNormalBuffer() const {return mNormalBuffer;}
};

class VolumeFusion
{
private:
    float* mFGpu;
    float* mWGpu;
    mutable float* mTgkGpu;
    
    MatrixGpu* mKdepth;
    
    int mSide;
    float mUnitsPerVoxel;
    
    float mBoundingBox[6];
    void initBoundingBox();
    
public:
    VolumeFusion(int side, float unitsPerVoxel, const double* Kdepth);
    ~VolumeFusion();
    
    void update(const float* depthGpu, const float* T);
    void raycast(const float* T);
    
    inline const float* getBoundingBox() const {return mBoundingBox;}
    float getMinimumDistanceTo(const float* point) const;
    float getMaximumDistanceTo(const float* point) const;
};

class FreenectFusion
{
private:
    int mWidth, mHeight;
    Measurement* mMeasurement;
    VolumeFusion* mVolume;
    float mLocation[16];
    
public:
    FreenectFusion(int width, int height,
                    const double* Kdepth, const double* Krgb);
    ~FreenectFusion();
    
    void update(void* depth);
    
    inline Measurement* getMeasurement() const {return mMeasurement;}
};

#endif // _FREENECTFUSION_H
