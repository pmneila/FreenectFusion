
#ifndef _FREENECTFUSION_H
#define _FREENECTFUSION_H

#include <stdint.h>

class float2;

class IntrinsicMatrix
{
private:
    float mK[9];
    float* mKGpu;
    float mKinv[9];
    float* mKinvGpu;
    
public:
    IntrinsicMatrix(const double* K);
    ~IntrinsicMatrix();
    
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
    
    IntrinsicMatrix mKdepth;
    
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

class FreenectFusion
{
private:
    int mWidth, mHeight;
    Measurement* mMeasurement;
    
public:
    FreenectFusion(int width, int height,
                    const double* Kdepth, const double* Krgb);
    ~FreenectFusion();
    
    void update(void* depth);
    
    inline Measurement* getMeasurement() const {return mMeasurement;}
};

#endif // _FREENECTFUSION_H
