
#ifndef _FREENECTFUSION_H
#define _FREENECTFUSION_H

#include <stdint.h>

class float2;
class float3;
class cudaArray;
class cudaMemcpy3DParms;

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
    
    inline const int* getMaskGpu() const {return mMaskGpu;}
    
    inline const MatrixGpu* getKdepth() const {return mKdepth;}
    
    inline unsigned int getGLVertexBuffer() const {return mVertexBuffer;}
    inline unsigned int getGLNormalBuffer() const {return mNormalBuffer;}
    
    inline int getWidth() const {return mWidth;}
    inline int getHeight() const {return mHeight;}
};

class VolumeFusion
{
private:
    float* mFGpu;
    float* mWGpu;
    mutable cudaArray* mFArray;
    cudaMemcpy3DParms* mCopyParams;
    
    int mSide;
    float mUnitsPerVoxel;
    
    float mBoundingBox[6];
    
    void initBoundingBox();
    void initFArray();
    
public:
    VolumeFusion(int side, float unitsPerVoxel);
    ~VolumeFusion();
    
    void update(const Measurement& measurement, const float* T);
    
    inline const float* getBoundingBox() const {return mBoundingBox;}
    float getMinimumDistanceTo(const float* point) const;
    float getMaximumDistanceTo(const float* point) const;
    
    inline int getSide() const {return mSide;}
    inline float getUnitsPerVoxel() const {return mUnitsPerVoxel;}
    
    inline const float* getFGpu() const {return mFGpu;}
    
    template<typename texture>
    void bindTextureToF(texture& tex) const;
};

class VolumeMeasurement
{
private:
    int mWidth, mHeight, mNumVertices;
    unsigned int mVertexBuffer, mNormalBuffer;
    MatrixGpu* mKdepth;
    float mT[16];
    
public:
    VolumeMeasurement(int width, int height, const double* Kdepth);
    ~VolumeMeasurement();
    
    void measure(const VolumeFusion& volume, const float* T);
    
    inline unsigned int getGLVertexBuffer() const {return mVertexBuffer;}
    inline unsigned int getGLNormalBuffer() const {return mNormalBuffer;}
    inline const float* getTransform() const {return mT;}
};

class Tracker
{
private:
    int mMaxNumVertices;
    float* mAAGpu;
    float* mAbGpu;
    float mAA[21];
    float mAb[6];
    mutable float* mCurrentTGpu;
    mutable float* mCurrent2InitTGpu;
    float mTrackTransform[16];
    
    void trackStep(float* AA, float* Ab, const float* currentT, const float* current2InitT,
                   float3* verticesOld, float3* normalsOld,
                   float3* verticesNew, float3* normalsNew,
                   const Measurement& meas, const VolumeMeasurement& volMeas);
    
    void solveSystem(float* incT, const float* AA, const float* Ab);
    
public:
    Tracker(int maxNumVertices);
    ~Tracker();
    void track(const Measurement& meas, const VolumeMeasurement& volMeas,
                const float* initT=0, float* res=0);
    inline const float* getTrackTranform() const {return mTrackTransform;}
};

class FreenectFusion
{
private:
    int mWidth, mHeight;
    Measurement* mMeasurement;
    VolumeFusion* mVolume;
    VolumeMeasurement* mVolumeMeasurement;
    Tracker* mTracker;
    float mLocation[16];
    
    bool mActiveTracking;
    
public:
    FreenectFusion(int width, int height,
                    const double* Kdepth, const double* Krgb);
    ~FreenectFusion();
    
    void update(void* depth);
    
    inline Measurement* getMeasurement() const {return mMeasurement;}
    inline VolumeFusion* getVolume() const {return mVolume;}
    inline VolumeMeasurement* getVolumeMeasurement() const {return mVolumeMeasurement;}
    inline const float* getLocation() const {return mLocation;}
    
    inline void toggleTracking() {mActiveTracking ^= 1;}
    inline void setTracking(bool on) {mActiveTracking = on;}
    inline bool isTracking() const {return mActiveTracking;}
};

#endif // _FREENECTFUSION_H
