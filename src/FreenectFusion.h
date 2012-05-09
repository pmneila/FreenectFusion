
#ifndef _FREENECTFUSION_H
#define _FREENECTFUSION_H

#include <stdint.h>

class float2;

class Measurement
{
private:
    int mWidth, mHeight, mNumElements;
    float* mDepthGpu;
    uint16_t* mRawDepthGpu;
    
    mutable float* mDepth;
    float2* mPixelPositionsGpu;
    
    void initPixelPositions();
    
public:
    Measurement(int width, int height);
    ~Measurement();
    
    void setDepth(uint16_t* depth);
    
    inline float* getDepthGpu() {return mDepth;}
    inline const float* getDepthGpu() const {return mDepth;}
    const float* getDepthHost() const;
};

class FreenectFusion
{
private:
    int mWidth, mHeight;
    Measurement* mMeasurement;
    
public:
    FreenectFusion(int width, int height,
                    const double* K_depth, const double* K_rgb);
    ~FreenectFusion();
    
    void update(void* depth);
    
    inline Measurement* getMeasurement() const {return mMeasurement;}
};

#endif // _FREENECTFUSION_H
