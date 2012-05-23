
#include "FreenectFusion.h"

#include "glheaders.h"
#include "cudautils.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdexcept>
#include <algorithm>

#include <cmath>
#include <numeric>
#include <iostream>

#define ISPOW2(x) !((x)&((x)-1))

MatrixGpu::MatrixGpu(int side, const double* K, const double* Kinv)
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

MatrixGpu* MatrixGpu::newIntrinsicMatrix(const double* K)
{
    double Kinv[9];
    std::fill(Kinv, Kinv+9, 0.0);
    Kinv[0] = 1.0/K[0];
    Kinv[4] = 1.0/K[4];
    Kinv[2] = -K[2]/K[0];
    Kinv[5] = -K[5]/K[4];
    Kinv[8] = 1.0;
    return new MatrixGpu(3, K, Kinv);
}

MatrixGpu* MatrixGpu::newTransformMatrix(const double* T)
{
    return 0;
}

Measurement::Measurement(int width, int height, const double* Kdepth)
    : mWidth(width), mHeight(height), mNumVertices(width*height)
{
    mKdepth = MatrixGpu::newIntrinsicMatrix(Kdepth);
    
    // Depth related data.
    size_t pitch;
    cudaSafeCall(cudaMallocPitch((void**)&mDepthGpu, &pitch, sizeof(float)*width, height));
    cudaSafeCall(cudaMallocPitch((void**)&mRawDepthGpu, &pitch, sizeof(uint16_t)*width, height));
    
    mDepth = new float[width * height];
    
    // Vertices and normals.
    glGenBuffers(1, &mVertexBuffer);
    glGenBuffers(1, &mNormalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, mNumVertices*12, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
    glBufferData(GL_ARRAY_BUFFER, mNumVertices*12, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(mVertexBuffer);
    cudaGLRegisterBufferObject(mNormalBuffer);
    cudaSafeCall(cudaMallocPitch((void**)&mMaskGpu, &pitch, sizeof(int)*width, height));
}

Measurement::~Measurement()
{
    delete mKdepth;
    cudaSafeCall(cudaFree(mDepthGpu));
    cudaSafeCall(cudaFree(mRawDepthGpu));
    delete [] mDepth;
    
    glDeleteBuffers(1, &mVertexBuffer);
    glDeleteBuffers(1, &mNormalBuffer);
    cudaSafeCall(cudaFree(mMaskGpu));
}

const float* Measurement::getDepthHost() const
{
    cudaSafeCall(cudaMemcpy(mDepth, mDepthGpu, sizeof(float)*mNumVertices,
            cudaMemcpyDeviceToHost));
    return mDepth;
}

VolumeFusion::VolumeFusion(int side, float unitsPerVoxel)
    : mSide(side), mUnitsPerVoxel(unitsPerVoxel)
{
    if(!ISPOW2(mSide))
        throw std::runtime_error("side must be power of 2");
    
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

VolumeMeasurement::VolumeMeasurement(int width, int height, const double* Kdepth)
    : mWidth(width), mHeight(height), mNumVertices(mWidth*mHeight)
{
    mKdepth = MatrixGpu::newIntrinsicMatrix(Kdepth);
    
    // Vertices and normals.
    glGenBuffers(1, &mVertexBuffer);
    glGenBuffers(1, &mNormalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, mNumVertices*12, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
    glBufferData(GL_ARRAY_BUFFER, mNumVertices*12, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(mVertexBuffer);
    cudaGLRegisterBufferObject(mNormalBuffer);
}

VolumeMeasurement::~VolumeMeasurement()
{
    delete mKdepth;
    glDeleteBuffers(1, &mVertexBuffer);
    glDeleteBuffers(1, &mNormalBuffer);
}

FreenectFusion::FreenectFusion(int width, int height,
                        const double* Kdepth, const double* Krgb)
    : mWidth(width), mHeight(height)
{
    static const float initLocation[16] = {1.f, 0.f, 0.f, 0.f,
                                           0.f, 1.f, 0.f, 0.f,
                                           0.f, 0.f, 1.f, -1000.f,
                                           0.f, 0.f, 0.f, 1.f};
    
    mMeasurement = new Measurement(width, height, Kdepth);
    mVolume = new VolumeFusion(128, 7.8125f);
    mVolumeMeasurement = new VolumeMeasurement(width, height, Kdepth);
    std::copy(initLocation, initLocation+16, mLocation);
}

FreenectFusion::~FreenectFusion()
{
    delete mMeasurement;
    delete mVolume;
    delete mVolumeMeasurement;
}

void FreenectFusion::update(void* depth)
{
    mMeasurement->setDepth((uint16_t*)depth);
    mVolume->update(*mMeasurement, mLocation);
    mVolumeMeasurement->measure(*mVolume, mLocation);
}
