
#include "FreenectFusion.h"

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

Measurement::Measurement(int width, int height)
    : mWidth(width), mHeight(height), mNumElements(width*height)
{
    size_t pitch;
    cudaMallocPitch((void**)&mDepthGpu, &pitch, sizeof(float)*width, height);
    cudaMallocPitch((void**)&mRawDepthGpu, &pitch, sizeof(uint16_t)*width, height);
    
    mDepth = new float[width * height];
    
    initPixelPositions();
}

Measurement::~Measurement()
{
    cudaFree(mDepthGpu);
    cudaFree(mRawDepthGpu);
    delete [] mDepth;
    
    cudaFree(mPixelPositionsGpu);
}

void Measurement::initPixelPositions()
{
    size_t pitch;
    float2* aux = new float2[mNumElements];
    for(int i=0; i<mHeight; ++i)
    {
        int base = i*mWidth;
        for(int j=0; j<mWidth; ++j)
        {
            aux[base + j].x = j;
            aux[base + j].y = i;
        }
    }
    cudaMallocPitch((void**)&mPixelPositionsGpu, &pitch, sizeof(float2)*mWidth, mHeight);
    cudaMemcpy(mPixelPositionsGpu, aux, sizeof(float2)*mNumElements,
            cudaMemcpyHostToDevice);
}

const float* Measurement::getDepthHost() const
{
    cudaMemcpy(mDepth, mDepthGpu, sizeof(float)*mNumElements,
            cudaMemcpyDeviceToHost);
    return mDepth;
}

FreenectFusion::FreenectFusion(int width, int height,
                        const double* K_depth, const double* K_rgb)
    : mWidth(width), mHeight(height)
{
    mMeasurement = new Measurement(width, height);
}

FreenectFusion::~FreenectFusion()
{
    delete mMeasurement;
}

void FreenectFusion::update(void* depth)
{
    mMeasurement->setDepth((uint16_t*)depth);
}
