
#include "FreenectFusion.h"

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

IntrinsicMatrix::IntrinsicMatrix(const double* K)
{
    std::copy(K, K+9, mK);
    cudaMalloc((void**)&mKGpu, sizeof(float)*9);
    cudaMemcpy(mKGpu, mK, sizeof(float)*9, cudaMemcpyHostToDevice);
    
    std::fill(mKinv, mKinv+9, 0.f);
    mKinv[0] = 1.f/mK[0];
    mKinv[4] = 1.f/mK[4];
    mKinv[2] = -mK[2]/mK[0];
    mKinv[5] = -mK[5]/mK[4];
    cudaMalloc((void**)&mKinvGpu, sizeof(float)*9);
    cudaMemcpy(mKinvGpu, mKinv, sizeof(float)*9, cudaMemcpyHostToDevice);
}

IntrinsicMatrix::~IntrinsicMatrix()
{
    cudaFree(mKGpu);
    cudaFree(mKinvGpu);
}

Measurement::Measurement(int width, int height, const double* Kdepth)
    : mWidth(width), mHeight(height), mNumVertices(width*height),
    mKdepth(Kdepth)
{
    // Depth related data.
    size_t pitch;
    cudaMallocPitch((void**)&mDepthGpu, &pitch, sizeof(float)*width, height);
    cudaMallocPitch((void**)&mRawDepthGpu, &pitch, sizeof(uint16_t)*width, height);
    
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
    cudaMallocPitch((void**)&mMaskGpu, &pitch, sizeof(int)*width, height);
}

Measurement::~Measurement()
{
    cudaFree(mDepthGpu);
    cudaFree(mRawDepthGpu);
    delete [] mDepth;
    
    glDeleteBuffers(1, &mVertexBuffer);
    glDeleteBuffers(1, &mNormalBuffer);
    cudaFree(mMaskGpu);
}

const float* Measurement::getDepthHost() const
{
    cudaMemcpy(mDepth, mDepthGpu, sizeof(float)*mNumVertices,
            cudaMemcpyDeviceToHost);
    return mDepth;
}

FreenectFusion::FreenectFusion(int width, int height,
                        const double* Kdepth, const double* Krgb)
    : mWidth(width), mHeight(height)
{
    mMeasurement = new Measurement(width, height, Kdepth);
}

FreenectFusion::~FreenectFusion()
{
    delete mMeasurement;
}

void FreenectFusion::update(void* depth)
{
    mMeasurement->setDepth((uint16_t*)depth);
}
