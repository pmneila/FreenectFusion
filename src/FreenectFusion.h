
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
    static MatrixGpu* newIntrinsicMatrix(const float* K);
    static MatrixGpu* newTransformMatrix(const float* T);
    
    MatrixGpu(int side, const float* K, const float* Kinv=0);
    ~MatrixGpu();
    
    inline operator const float*() const {return mK;}
    
    inline const float* get() const {return mK;}
    inline const float* getInverse() const {return mKinv;}
    inline const float* getGpu() const {return mKGpu;}
    inline const float* getInverseGpu() const {return mKinvGpu;}
};

class Measurement;

class PyramidMeasurement
{
private:
    Measurement* mParent;
    PyramidMeasurement* mParent2;
    int mLevel;
    int mNumVertices;
    int mWidth, mHeight;
    float* mDepthGpu;
    int* mMaskGpu;
    unsigned int mVertexBuffer;
    unsigned int mNormalBuffer;
    
    float mK[9];
    float mKInv[9];
    
    void initBuffers();
    void initK(float a, float b);
    
    void update1();
    void update2();
    
public:
    PyramidMeasurement(Measurement* parent);
    PyramidMeasurement(PyramidMeasurement* parent);
    ~PyramidMeasurement();
    
    void update();
    
    inline unsigned int getGLVertexBuffer() const {return mVertexBuffer;}
    inline unsigned int getGLNormalBuffer() const {return mNormalBuffer;}
    
    inline const float* getDepthGpu() const {return mDepthGpu;}
    
    inline const float* getK() const {return mK;}
    inline const float* getKInverse() const {return mKInv;}
    inline int getLevel() const {return mLevel;}
    inline int getWidth() const {return mWidth;}
    inline int getHeight() const {return mHeight;}
    inline int getNumVertices() const {return mNumVertices;}
};

class Measurement
{
private:
    int mWidth, mHeight, mNumVertices;
    float* mDepthGpu;
    uint16_t* mRawDepthGpu;
    mutable float* mDepth;
    
    MatrixGpu* mKdepth;
    
    PyramidMeasurement* mPyramid[3];
    
public:
    Measurement(int width, int height, const float* Kdepth);
    ~Measurement();
    
    void setDepth(uint16_t* depth);
    
    inline float* getDepthGpu() {return mDepthGpu;}
    inline const float* getDepthGpu() const {return mDepthGpu;}
    const float* getDepthHost() const;
    
    inline const float* getK() const {return mKdepth->get();}
    inline const float* getKInverse() const {return mKdepth->getInverse();}
    
    inline unsigned int getGLVertexBuffer(int level=0) const
    { return mPyramid[level]->getGLVertexBuffer(); }
    
    inline unsigned int getGLNormalBuffer(int level=0) const
    { return mPyramid[level]->getGLNormalBuffer(); }
    
    inline int getWidth() const {return mWidth;}
    inline int getHeight() const {return mHeight;}
    
    const PyramidMeasurement* getLevel(int level) const {return mPyramid[level];}
};

class VolumeFusion
{
private:
    float* mFGpu;
    float* mWGpu;
    mutable cudaArray* mFArray;
    cudaMemcpy3DParms* mCopyParams;
    
    int mSideLog, mSide;
    float mUnitsPerVoxel;
    
    float mBoundingBox[6];
    
    void initBoundingBox();
    void initFArray();
    
public:
    VolumeFusion(int sidelog, float unitsPerVoxel);
    ~VolumeFusion();
    
    void update(const Measurement& measurement, const float* T);
    
    inline const float* getBoundingBox() const {return mBoundingBox;}
    float getMinimumDistanceTo(const float* point) const;
    float getMaximumDistanceTo(const float* point) const;
    
    inline int getSide() const {return mSide;}
    inline int getSideLog() const {return mSideLog;}
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
    VolumeMeasurement(int width, int height, const float* Kdepth);
    ~VolumeMeasurement();
    
    void measure(const VolumeFusion& volume, const float* T);
    
    inline unsigned int getGLVertexBuffer() const {return mVertexBuffer;}
    inline unsigned int getGLNormalBuffer() const {return mNormalBuffer;}
    inline const float* getTransform() const {return mT;}
    
    inline int getWidth() const {return mWidth;}
    inline int getHeight() const {return mHeight;}
    inline int getNumVertices() const {return mNumVertices;}
    
    inline const float* getK() const {return mKdepth->get();}
    inline const float* getKInverse() const {return mKdepth->getInverse();}
};

class Tracker
{
private:
    int mMaxNumVertices;
    float* mAAGpu;
    float* mAbGpu;
    float mAA[21];
    float mAb[6];
    float mTrackTransform[16];
    float3* mVertexCorrespondencesGpu;
    float3* mNormalCorrespondencesGpu;
    
    void searchCorrespondences(float3* vertexCorresp, float3* normalsCorresp,
                   const float* K,
                   const float* currentT, const float* current2InitT,
                   const float3* verticesOld, const float3* normalsOld,
                   const float3* verticesNew, const float3* normalsNew,
                   int widthOld, int heightOld, int widthNew, int heightNew);
    void trackStep(float* AA, float* Ab, const float* currentT,
                   const float3* verticesMeasure, const float3* normalsMeasure,
                   const float3* verticesCorresp, const float3* normalsCorresp,
                   int numVertices);
    
    void solveSystem(float* incT, const float* AA, const float* Ab);
    
public:
    Tracker(int maxNumVertices);
    ~Tracker();
    void track(const Measurement& meas, const VolumeMeasurement& volMeas,
                const float* initT=0, float* res=0);
    inline const float* getTrackTransform() const {return mTrackTransform;}
};

class MarchingCubes;

class FreenectFusion
{
private:
    int mWidth, mHeight;
    Measurement* mMeasurement;
    VolumeFusion* mVolume;
    VolumeMeasurement* mVolumeMeasurement;
    Tracker* mTracker;
    MarchingCubes* mMC;
    
    float mLocation[16];
    
    bool mActiveTracking;
    bool mActiveUpdate;
    
public:
    FreenectFusion(int width, int height,
                    const float* Kdepth, const float* Krgb);
    ~FreenectFusion();
    
    void update(void* depth);
    
    inline Measurement* getMeasurement() const {return mMeasurement;}
    inline VolumeFusion* getVolume() const {return mVolume;}
    inline VolumeMeasurement* getVolumeMeasurement() const {return mVolumeMeasurement;}
    inline MarchingCubes* getMarchingCubes() const {return mMC;}
    inline const float* getLocation() const {return mLocation;}
    
    inline void toggleUpdate() {mActiveUpdate ^= 1;}
    inline void setUpdate(bool on) {mActiveUpdate = on;}
    inline bool isUpdating() const {return mActiveUpdate;}
    
    inline void toggleTracking() {mActiveTracking ^= 1;}
    inline void setTracking(bool on) {mActiveTracking = on;}
    inline bool isTracking() const {return mActiveTracking;}
};

#endif // _FREENECTFUSION_H
