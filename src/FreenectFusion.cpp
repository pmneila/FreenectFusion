
#include "FreenectFusion.h"

#include "MarchingCubes.h"
#include "glheaders.h"
#include "cudautils.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdexcept>
#include <algorithm>

#include <cmath>
#include <numeric>
#include <iostream>

#include "Eigen/Dense"

#define ISPOW2(x) !((x)&((x)-1))

static void invertIntrinsics(float* res, const float* K)
{
    std::fill(res, res+9, 0.f);
    res[0] = 1.0/K[0];
    res[4] = 1.0/K[4];
    res[2] = -K[2]/K[0];
    res[5] = -K[5]/K[4];
    res[8] = 1.0;
}

static void invertTransform(float* res, const float* T)
{
    res[0] = T[0]; res[1] = T[4]; res[2] = T[8];
    res[4] = T[1]; res[5] = T[5]; res[6] = T[9];
    res[8] = T[2]; res[9] = T[6]; res[10] = T[10];
    res[3] = -(res[0]*T[3] + res[1]*T[7] + res[2]*T[11]);
    res[7] = -(res[4]*T[3] + res[5]*T[7] + res[6]*T[11]);
    res[11] = -(res[8]*T[3] + res[9]*T[7] + res[10]*T[11]);
    res[12] = res[13] = res[14] = 0.f;
    res[15] = 1.f;
}

static void multiplyTransforms(float* res, const float* T1, const float* T2)
{
    for(int i=0; i<4; ++i)
    {
        for(int j=0; j<4; ++j)
        {
            int offset = i*4+j;
            res[offset] = 0;
            for(int k=0; k<4; ++k)
            {
                res[offset] += T1[i*4+k]*T2[k*4+j];
            }
        }
    }
}

MatrixGpu::MatrixGpu(int side, const float* K, const float* Kinv)
    : mSide(side), mSize(side*side)
{
    mK = new float[mSize];
    std::copy(K, K+mSize, mK);
    cudaSafeCall(cudaMalloc((void**)&mKGpu, sizeof(float)*mSize));
    cudaSafeCall(cudaMemcpy(mKGpu, mK, sizeof(float)*mSize, cudaMemcpyHostToDevice));
    
    if(Kinv != 0)
    {
        mKinv = new float[mSize];
        std::copy(Kinv, Kinv+mSize, mKinv);
        cudaSafeCall(cudaMalloc((void**)&mKinvGpu, sizeof(float)*mSize));
        cudaSafeCall(cudaMemcpy(mKinvGpu, mKinv, sizeof(float)*mSize, cudaMemcpyHostToDevice));
    }
}

MatrixGpu::~MatrixGpu()
{
    delete [] mK;
    delete [] mKinv;
    cudaSafeCall(cudaFree(mKGpu));
    cudaSafeCall(cudaFree(mKinvGpu));
}

MatrixGpu* MatrixGpu::newIntrinsicMatrix(const float* K)
{
    float Kinv[9];
    std::fill(Kinv, Kinv+9, 0.0);
    Kinv[0] = 1.0/K[0];
    Kinv[4] = 1.0/K[4];
    Kinv[2] = -K[2]/K[0];
    Kinv[5] = -K[5]/K[4];
    Kinv[8] = 1.0;
    return new MatrixGpu(3, K, Kinv);
}

MatrixGpu* MatrixGpu::newTransformMatrix(const float* T)
{
    return 0;
}

Measurement::Measurement(int width, int height, const float* Kdepth)
    : mWidth(width), mHeight(height), mNumVertices(width*height)
{
    mKdepth = MatrixGpu::newIntrinsicMatrix(Kdepth);
    
    // Depth related data.
    size_t pitch;
    cudaSafeCall(cudaMallocPitch((void**)&mDepthGpu, &pitch, sizeof(float)*width, height));
    cudaSafeCall(cudaMallocPitch((void**)&mRawDepthGpu, &pitch, sizeof(uint16_t)*width, height));
    
    mDepth = new float[width * height];
    
    mPyramid[0] = new PyramidMeasurement(this);
    mPyramid[1] = new PyramidMeasurement(mPyramid[0]);
    mPyramid[2] = new PyramidMeasurement(mPyramid[1]);
}

Measurement::~Measurement()
{
    delete mPyramid[0];
    delete mPyramid[1];
    delete mPyramid[2];
    
    delete mKdepth;
    cudaSafeCall(cudaFree(mDepthGpu));
    cudaSafeCall(cudaFree(mRawDepthGpu));
    delete [] mDepth;
}

const float* Measurement::getDepthHost() const
{
    cudaSafeCall(cudaMemcpy(mDepth, mDepthGpu, sizeof(float)*mNumVertices,
            cudaMemcpyDeviceToHost));
    return mDepth;
}

PyramidMeasurement::PyramidMeasurement(Measurement* parent)
    : mParent(parent), mParent2(0), mLevel(0)
{
    mWidth = mParent->getWidth();
    mHeight = mParent->getHeight();
    mNumVertices = mWidth * mHeight;
    
    std::copy(parent->getK(), parent->getK()+9, mK);
    invertIntrinsics(mKInv, mK);
    initBuffers();
}

PyramidMeasurement::PyramidMeasurement(PyramidMeasurement* parent)
    : mParent(0), mParent2(parent), mLevel(parent->getLevel()+1)
{
    mWidth = mParent2->getWidth()>>1;
    mHeight = mParent2->getHeight()>>1;
    mNumVertices = mWidth * mHeight;
    
    std::copy(parent->getK(), parent->getK()+9, mK);
    initK(0.5f, -0.25f);
    initBuffers();    
}

void PyramidMeasurement::initK(float a, float b)
{
    // Create the new intrinsic matrix.
    mK[0] *= a;
    mK[4] *= a;
    mK[2] = mK[2]*a + b;
    mK[5] = mK[5]*a + b;
    // And its inverse.
    invertIntrinsics(mKInv, mK);
}

void PyramidMeasurement::initBuffers()
{
    // Create the local depth map.
    cudaSafeCall(cudaMalloc((void**)&mDepthGpu, mNumVertices*sizeof(float)));
    // Create the local depth map.
    cudaSafeCall(cudaMalloc((void**)&mMaskGpu, mNumVertices*sizeof(int)));
    
    // Create the vertex and normal buffers.
    glGenBuffers(1, &mVertexBuffer);
    glGenBuffers(1, &mNormalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, mNumVertices*3*sizeof(float), 0, GL_DYNAMIC_COPY);
    glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
    glBufferData(GL_ARRAY_BUFFER, mNumVertices*3*sizeof(float), 0, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(mVertexBuffer);
    cudaGLRegisterBufferObject(mNormalBuffer);
}

PyramidMeasurement::~PyramidMeasurement()
{
    cudaSafeCall(cudaFree(mDepthGpu));
    cudaSafeCall(cudaFree(mMaskGpu));
    cudaSafeCall(cudaGLUnregisterBufferObject(mVertexBuffer));
    cudaSafeCall(cudaGLUnregisterBufferObject(mNormalBuffer));
    glDeleteBuffers(1, &mVertexBuffer);
    glDeleteBuffers(1, &mNormalBuffer);
}

VolumeFusion::VolumeFusion(int sidelog, float unitsPerVoxel)
    : mSideLog(sidelog), mSide(1<<sidelog), mUnitsPerVoxel(unitsPerVoxel)
{
    if(!ISPOW2(mSide) || mSide < 8)
        throw std::runtime_error("side must be power of 2 and greater or equal to 8");
    
    unsigned int numElements = mSide*mSide*mSide;
    cudaSafeCall(cudaMalloc((void**)&mFGpu, numElements*sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)&mWGpu, numElements*sizeof(float)));
    
    // Fill with zeros.
    float* zero = new float[numElements];
    std::fill(zero, zero+numElements, 0.f);
    cudaSafeCall(cudaMemcpy(mFGpu, zero, numElements*sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(mWGpu, zero, numElements*sizeof(float), cudaMemcpyHostToDevice));
    delete [] zero;
    
    initBoundingBox();
    initFArray();
}

VolumeFusion::~VolumeFusion()
{
    cudaSafeCall(cudaFree(mFGpu));
    cudaSafeCall(cudaFree(mWGpu));
    cudaSafeCall(cudaFreeArray(mFArray));
}

void VolumeFusion::initFArray()
{
    static const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    
    cudaExtent extent = make_cudaExtent(mSide, mSide, mSide);
    cudaSafeCall(cudaMalloc3DArray(&mFArray, &channelDesc, extent));
    
    // Create the copy params.
    mCopyParams = new cudaMemcpy3DParms;
    mCopyParams->srcPtr = make_cudaPitchedPtr((void*)mFGpu,
                                            sizeof(float)*mSide, mSide, mSide);
    mCopyParams->dstArray = mFArray;
    mCopyParams->extent = make_cudaExtent(mSide, mSide, mSide);
    mCopyParams->kind = cudaMemcpyDeviceToDevice;
}

float VolumeFusion::getMinimumDistanceTo(const float* point) const
{
    static const float center[] = {0.f, 0.f, 0.f};
    float point2[3];
    float point_bbox[3];
    float v[3];
    int size = mSide*mUnitsPerVoxel / 2;
    // point2 = point - center
    std::transform(point, point+3, center, point2, std::minus<float>());
    std::copy(point2, point2+3, point_bbox);
    std::replace_if(point_bbox, point_bbox+3, std::bind2nd(std::less<float>(),-size), -size);
    std::replace_if(point_bbox, point_bbox+3, std::bind2nd(std::greater<float>(),size), size);
    std::transform(point_bbox, point_bbox+3, point2, v, std::minus<float>());
    return std::sqrt(std::inner_product(v, v+3, v, 0));
}

float VolumeFusion::getMaximumDistanceTo(const float* point) const
{
    float corner[3];
    float distance[8];
    float v[3];
    for(int i=0; i<8; ++i)
    {
        corner[0] = i&1 ? mBoundingBox[3] : -mBoundingBox[0];
        corner[1] = i&2 ? mBoundingBox[4] : -mBoundingBox[1];
        corner[2] = i&4 ? mBoundingBox[5] : -mBoundingBox[2];
        std::transform(point, point+3, corner, v, std::minus<float>());
        distance[i] = std::sqrt(std::inner_product(v, v+3, v, 0));
    }
    return *std::max_element(distance, distance+8);
}

VolumeMeasurement::VolumeMeasurement(int width, int height, const float* Kdepth)
    : mWidth(width), mHeight(height), mNumVertices(mWidth*mHeight)
{
    mKdepth = MatrixGpu::newIntrinsicMatrix(Kdepth);
    
    // Vertices and normals.
    glGenBuffers(1, &mVertexBuffer);
    glGenBuffers(1, &mNormalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, mNumVertices*3*sizeof(float), NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
    glBufferData(GL_ARRAY_BUFFER, mNumVertices*3*sizeof(float), NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(mVertexBuffer);
    cudaGLRegisterBufferObject(mNormalBuffer);
}

VolumeMeasurement::~VolumeMeasurement()
{
    delete mKdepth;
    cudaSafeCall(cudaGLUnregisterBufferObject(mVertexBuffer));
    cudaSafeCall(cudaGLUnregisterBufferObject(mNormalBuffer));
    glDeleteBuffers(1, &mVertexBuffer);
    glDeleteBuffers(1, &mNormalBuffer);
}

Tracker::Tracker(int maxNumVertices)
    : mMaxNumVertices(maxNumVertices)
{
    cudaSafeCall(cudaMalloc((void**)&mAAGpu, sizeof(float)*maxNumVertices*21));
    cudaSafeCall(cudaMalloc((void**)&mAbGpu, sizeof(float)*maxNumVertices*6));
    
    cudaSafeCall(cudaMalloc((void**)&mVertexCorrespondencesGpu, sizeof(float3)*maxNumVertices));
    cudaSafeCall(cudaMalloc((void**)&mNormalCorrespondencesGpu, sizeof(float3)*maxNumVertices));
}

Tracker::~Tracker()
{
    cudaSafeCall(cudaFree(mAAGpu));
    cudaSafeCall(cudaFree(mAbGpu));
    
    cudaSafeCall(cudaFree(mVertexCorrespondencesGpu));
    cudaSafeCall(cudaFree(mNormalCorrespondencesGpu));
}

void Tracker::track(const Measurement& meas, const VolumeMeasurement& volMeas,
                    const float* initT, float* res)
{
    // TODO: Asserts of shapes.
    if(initT == 0)
        initT = volMeas.getTransform();
    
    float AA[21], Ab[16];
    float currentT[16], currentT2[16];
    float incT[16];
    float current2InitT[16] = {1.f, 0.f, 0.f, 0.f,
                               0.f, 1.f, 0.f, 0.f,
                               0.f, 0.f, 1.f, 0.f,
                               0.f, 0.f, 0.f, 1.f};
    float initTinv[16];
    std::copy(initT, initT+16, currentT);
    invertTransform(initTinv, initT);
    
    float3* verticesMeasure;
    float3* normalsMeasure;
    float3* verticesVolume;
    float3* normalsVolume;
    
    // TODO: Are these heavy calls?
    cudaGLMapBufferObject((void**)&verticesVolume, volMeas.getGLVertexBuffer());
    cudaGLMapBufferObject((void**)&normalsVolume, volMeas.getGLNormalBuffer());
    int widthRaycast = volMeas.getWidth();
    int heightRaycast = volMeas.getHeight();
    
    for(int level=2; level>=0; --level)
    {
        const PyramidMeasurement* pyr = meas.getLevel(level);
        
        cudaGLMapBufferObject((void**)&verticesMeasure, pyr->getGLVertexBuffer());
        cudaGLMapBufferObject((void**)&normalsMeasure, pyr->getGLNormalBuffer());
        int widthMeasure = pyr->getWidth();
        int heightMeasure = pyr->getHeight();
        
        for(int i=0; i<3; ++i)
        {
            multiplyTransforms(current2InitT, initTinv, currentT);
            searchCorrespondences(mVertexCorrespondencesGpu, mNormalCorrespondencesGpu,
                                volMeas.getK(), currentT, current2InitT,
                                verticesMeasure, normalsMeasure,
                                verticesVolume, normalsVolume,
                                widthMeasure, heightMeasure, widthRaycast, heightRaycast);
            trackStep(AA, Ab, currentT,
                      verticesMeasure, normalsMeasure,
                      mVertexCorrespondencesGpu, mNormalCorrespondencesGpu,
                      pyr->getNumVertices());
            solveSystem(incT, AA, Ab);
            multiplyTransforms(currentT2, incT, currentT);
            std::copy(currentT2, currentT2+16, currentT);
        }
        
        cudaGLUnmapBufferObject(pyr->getGLVertexBuffer());
        cudaGLUnmapBufferObject(pyr->getGLNormalBuffer());
    }
    
    cudaGLUnmapBufferObject(volMeas.getGLVertexBuffer());
    cudaGLUnmapBufferObject(volMeas.getGLNormalBuffer());
    
    if(res!=0)
        std::copy(currentT, currentT+16, res);
    
    std::copy(currentT, currentT+16, mTrackTransform);
}

void Tracker::solveSystem(float* incT, const float* AA, const float* Ab)
{
    typedef Eigen::Matrix<float, 6, 6> Matrix66f;
    typedef Eigen::Matrix<float, 6, 1> Vector6f;
    typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Matrix4fr;
    Matrix66f matAA;
    Vector6f matAb;
    Matrix4fr res = Eigen::Matrix4f::Identity();
    
    // Copy the input matrices into Eigen::Matrix instances.
    for(int i=0; i<6; ++i)
        for(int j=0; j<6; ++j)
        {
            if(i<=j)
                matAA(i,j) = AA[((11 - i)*i)/2 + j];//[i*6 - ((i*(i+1))/2) + j];
            else
                matAA(i,j) = AA[((11 - j)*j)/2 + i];
        }
    std::copy(Ab, Ab+6, matAb.data());
    // Solve the system.
    Vector6f aux = matAA.ldlt().solve(matAb);
    
    if(!std::isnan(aux(0)))
    {
        // Build the rotation matrix.
        Eigen::Matrix3f rot;
        rot << 1, aux(2), -aux(1),
               -aux(2), 1, aux(0),
               aux(1), -aux(0), 1;
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(rot, Eigen::ComputeFullU | Eigen::ComputeFullV);
        rot = svd.matrixU() * svd.matrixV().transpose();
    
        // Buld the transformation matrix.
        res.block<3,3>(0,0) = rot;
        res.block<3,1>(0,3) = aux.block<3,1>(3,0);
    }
    
    //std::cout << res << std::endl << std::endl;
    std::copy(res.data(), res.data()+16, incT);
}

FreenectFusion::FreenectFusion(int width, int height,
                        const float* Kdepth, const float* Krgb)
    : mWidth(width), mHeight(height), mActiveTracking(false),
    mActiveUpdate(true)
{
    static const float initLocation[16] = {1.f, 0.f, 0.f, 0.f,
                                           0.f, 1.f, 0.f, 0.f,
                                           0.f, 0.f, 1.f, -1000.f,
                                           0.f, 0.f, 0.f, 1.f};
    
    mMeasurement = new Measurement(width, height, Kdepth);
    mVolume = new VolumeFusion(7, /*7.8125f*/5.859375f);
    mVolumeMeasurement = new VolumeMeasurement(width, height, Kdepth);
    mTracker = new Tracker(width*height);
    //mMC = new MarchingCubes(mVolume);
    std::copy(initLocation, initLocation+16, mLocation);
}

FreenectFusion::~FreenectFusion()
{
    delete mMeasurement;
    delete mVolume;
    delete mVolumeMeasurement;
    delete mTracker;
    //delete mMC;
}

void FreenectFusion::update(void* depth)
{
    mMeasurement->setDepth((uint16_t*)depth);
    
    if(mActiveUpdate)
        mVolume->update(*mMeasurement, mLocation);
    
    mVolumeMeasurement->measure(*mVolume, mLocation);
    if(mActiveTracking)
    {
        mTracker->track(*mMeasurement, *mVolumeMeasurement);
        const float* newT = mTracker->getTrackTransform();
        std::copy(newT, newT+16, mLocation);
    }
    //mMC->computeMC();
}
