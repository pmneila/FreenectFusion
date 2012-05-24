
#include "FreenectFusion.h"

#include "cudautils.h"

#include <cuda_gl_interop.h>
#include <thrust/transform.h>
#include <thrust/fill.h>

texture<float, 2, cudaReadModeElementType> depth_texture;
texture<float, 2, cudaReadModeElementType> smooth_depth_texture;
texture<float, 3, cudaReadModeElementType> F_texture;

__constant__ float K[9];
__constant__ float invK[9];
__constant__ float Tgk[16];
__constant__ float Tk_1k[16];

inline __device__ float length2(float2 v)
{
    return sqrtf(v.x*v.x + v.y*v.y);
}

inline __device__ float dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

inline __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float3 transform3(const float* matrix, float3 v)
{
    float3 res;
    res.x = matrix[0]*v.x + matrix[1]*v.y + matrix[2]*v.z;
    res.y = matrix[3]*v.x + matrix[4]*v.y + matrix[5]*v.z;
    res.z = matrix[6]*v.x + matrix[7]*v.y + matrix[8]*v.z;
    return res;
}

__device__ float3 transform3_affine(const float* matrix, float3 v)
{
    float3 res;
    res.x = matrix[0]*v.x + matrix[1]*v.y + matrix[2]*v.z + matrix[3];
    res.y = matrix[4]*v.x + matrix[5]*v.y + matrix[6]*v.z + matrix[7];
    res.z = matrix[8]*v.x + matrix[9]*v.y + matrix[10]*v.z + matrix[11];
    return res;
}

__device__ float3 transform3_affine_inverse(const float* matrix, float3 v)
{
    float3 res;
    float3 v2 = make_float3(v.x-matrix[3], v.y-matrix[7], v.z-matrix[11]);
    res.x = matrix[0]*v2.x + matrix[4]*v2.y + matrix[8]*v2.z;
    res.y = matrix[1]*v2.x + matrix[5]*v2.y + matrix[9]*v2.z;
    res.z = matrix[2]*v2.x + matrix[6]*v2.y + matrix[10]*v2.z;
    return res;
}

inline __device__ float length(const float3& v)
{
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

__device__ float3 normalize(float3 v)
{
    float invLen = 1.0f / length(v);
    return invLen * v;
}

__device__ float gaussian(float t, float sigma)
{
    return exp(-t*t/(sigma*sigma));
}

__host__ __device__ float3 gridToWorld(float3 p, int side, float units_per_voxel)
{
    return make_float3((p.x - side/2) * units_per_voxel,
                        (p.y - side/2) * units_per_voxel,
                        (p.z - side/2) * units_per_voxel);
}

__host__ __device__ float3 worldToGrid(float3 p, int side, float units_per_voxel)
{
    return make_float3(p.x/units_per_voxel + side/2,
                        p.y/units_per_voxel + side/2,
                        p.z/units_per_voxel + side/2);
}

__global__ void compute_depth_2(float* depth, int width, int height, size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x >= width || y >= height)
        return;
    
    float* row = (float*)((char*)depth + y * pitch);
    
    if(row[x] == 2047)
    {
        row[x] = 0.f;
        return;
    }
    
    row[x] = 1000.f / (row[x] * -0.0030711016f + 3.3309495161f);
}

__global__ void compute_smooth_depth(float* smooth_depth,
                        int width, int height, size_t pitch,
                        float sigma1, float sigma2)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x >= width || y >= height)
        return;
    
    float* current_smooth_depth = (float*)((char*)smooth_depth + pitch*y) + x;
    
    float depth1 = tex2D(depth_texture, x, y);
    float cum = 0.f;
    float weight_cum = 0.f;
    for(int i=-5; i<=5; ++i)
        for(int j=-5; j<=5; ++j)
        {
            float depth2 = tex2D(depth_texture, x+i, y+j);
            float distance1 = length2(make_float2(i,j));
            float distance2 = depth1 - depth2;
            float weight1 = gaussian(distance1, sigma1);
            float weight2 = gaussian(distance2, sigma2);
            weight_cum += weight1 * weight2;
            cum += depth1 * weight1 * weight2;
        }
    cum /= weight_cum;
    *current_smooth_depth = cum;
}

/**
 * Generate vertices and normals from a depth stored in depth_texture.
 */
__global__ void measure(float3* vertices, float3* normals, int* mask,
                        int width, int height, size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int thid = width*y + x;
    
    if(x >= width || y >= height)
        return;
    
    float3* current_vertex = (float3*)((char*)vertices + pitch*y) + x;
    float3* current_normal = (float3*)((char*)normals + pitch*y) + x;
    
    float3 u = make_float3(float(x), float(y), 1.f);
    float3 v = make_float3(float(x+1), float(y), 1.f);
    float3 w = make_float3(float(x), float(y+1), 1.f);
    float depth = tex2D(depth_texture, x, y);
    u = depth * transform3(invK, u);
    v = tex2D(depth_texture, x+1, y) * transform3(invK, v);
    w = tex2D(depth_texture, x, y+1) * transform3(invK, w);
    
    float3 n = normalize(cross(v - u, w - u));
    *current_vertex = u;
    *current_normal = n;
    mask[thid] = depth > 0.01f;
}

__global__ void update_reconstruction(float* F, float* W,
                        int side, float units_per_voxel,
                        float mu, int init_slice)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z + init_slice;
    
    float* current_F = F + k*side*side + j*side + i;
    float* current_W = W + k*side*side + j*side + i;
    
    // Point 3D.
    float3 p = gridToWorld(make_float3(i,j,k), side, units_per_voxel);
    
    // Project the point.
    float3 x = transform3(K, transform3_affine_inverse(Tgk, p));
    x.x = round(x.x/x.z);
    x.y = round(x.y/x.z);
    x.z = 1.f;
    
    // Determine lambda.
    float3 aux = transform3(invK, x);
    float lambda = length(aux);
    
    float R = tex2D(depth_texture, x.x, x.y);
    
    float3 tgk = make_float3(Tgk[3], Tgk[7], Tgk[11]);
    float eta = R - length(tgk - p)/lambda;
    float F_rk = fminf(1.f, eta/mu);
    float W_rk = 1.f;
    if(F_rk < -1.f || R == 0.f)
        return;
    
    if(*current_F < -2.f)
        *current_F = F_rk;
    else
        *current_F = (*current_W * *current_F + W_rk * F_rk)/(*current_W + W_rk);
    *current_W = min(*current_W + W_rk, 10.f);
}

__global__ void raycast(float3* vertices, float3* normals,
                        int width, int height, size_t pitch,
                        int side, float units_per_voxel, float mu,
                        float mindistance, float maxdistance)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x >= width || y >= height)
        return;
    
    float3* current_vertex = (float3*)((char*)vertices + pitch*y) + x;
    float3* current_normal = (float3*)((char*)normals + pitch*y) + x;
    
    float3 ray = normalize(transform3(invK, make_float3(float(x), float(y), 1.f)));
    float3 tgk = make_float3(Tgk[3], Tgk[7], Tgk[11]);
    ray = transform3_affine(Tgk, ray) - tgk;
    
    *current_normal = make_float3(1.f, 1.f, 1.f);
    
    float step = 3.f*mu/4.f;
    float3 p = worldToGrid(tgk + mindistance * ray, side, units_per_voxel);
    float old_value = tex3D(F_texture, p.x, p.y, p.z);
    for(float distance = mindistance; distance < maxdistance; distance += step)
    {
        p = worldToGrid(tgk + distance * ray, side, units_per_voxel);
        float value = tex3D(F_texture, p.x, p.y, p.z);
        
        if(value < -2 || (old_value < 0 && value > 0))
            break;
        if(old_value > 0 && value <= 0)
        {
            float t = distance - step - (step * old_value)/(value - old_value);
            *current_vertex = tgk + t * ray;
            float valuex = tex3D(F_texture, p.x-1, p.y, p.z);
            float valuey = tex3D(F_texture, p.x, p.y-1, p.z);
            float valuez = tex3D(F_texture, p.x, p.y, p.z-1);
            *current_normal = normalize(make_float3(valuex-value, valuey-value, valuez-value));
            return;
        }
        
        old_value = value;
    }
    *current_vertex = make_float3(0.f, 0.f, 0.f);
}

__device__ float3 project(const float* K, const float* T, float3 point)
{
    return transform3(K, transform3_affine(T, point));
}

__device__ int2 hom2cart(float3 point)
{
    return make_int2(roundf(point.x/point.z), roundf(point.y/point.z));
}

__global__ void compute_tracking_matrices(float* AA, float* Ab,
                        float3* vertices_measure, float3* normals_measure,
                        float3* vertices_raycast, float3* normals_raycast,
                        int width, int height,
                        size_t AA_pitch, size_t Ab_pitch,
                        const int* mask,
                        float threshold_distance)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int thid = width*y + x;
    
    if(x >= width || y >= height)
        return;
    
    float* current_AA = (float*)((char*)AA + AA_pitch * thid);
    float* current_Ab = (float*)((char*)Ab + Ab_pitch * thid);
    
    float3 vertex_measure = vertices_measure[thid];
    
    // Get the corresponding pixel in the raycast image.
    int2 u_raycast = hom2cart(project(K, Tk_1k, vertex_measure));
    
    if(u_raycast.x < 0 || u_raycast.y < 0 ||
            u_raycast.x >= width || u_raycast.y >= height)
        return;
    
    int id_raycast = width*u_raycast.y + u_raycast.x;
    float3 vertex_raycast = vertices_raycast[id_raycast];
    
    float3 v = transform3_affine(Tgk, vertex_measure);
    float3 vdiff = vertex_raycast - v;
    float vertex_distance = length(vdiff);
    
    // Prune invalid matches.
    if(!mask[thid] || vertex_distance > threshold_distance)
        return;
    
    normals_measure[thid] = make_float3(1,1,1);
    
    float3 n = normals_raycast[thid];
    float b = dot(vdiff, n);
    current_Ab[0] = b*(v.z*n.y - v.y*n.z);
    current_Ab[1] = b*(-v.z*n.x + v.x*n.z);
    current_Ab[2] = b*(v.y*n.x - v.x*n.y);
    current_Ab[3] = b*n.x;
    current_Ab[4] = b*n.y;
    current_Ab[5] = b*n.z;
    
    current_AA[0] = (v.z*n.y - v.y*n.z)*(v.z*n.y - v.y*n.z);
    current_AA[1] = (v.z*n.y - v.y*n.z)*(-v.z*n.x + v.x*n.z);
    current_AA[2] = (v.z*n.y - v.y*n.z)*(v.y*n.x - v.x*n.y);
    current_AA[3] = (v.z*n.y - v.y*n.z)*n.x;
    current_AA[4] = (v.z*n.y - v.y*n.z)*n.y;
    current_AA[5] = (v.z*n.y - v.y*n.z)*n.z;
    
    current_AA[6]  = (-v.z*n.x + v.x*n.z)*(-v.z*n.x + v.x*n.z);
    current_AA[7]  = (-v.z*n.x + v.x*n.z)*(v.y*n.x - v.x*n.y);
    current_AA[8]  = (-v.z*n.x + v.x*n.z)*n.x;
    current_AA[9]  = (-v.z*n.x + v.x*n.z)*n.y;
    current_AA[10] = (-v.z*n.x + v.x*n.z)*n.z;
    
    current_AA[11] = (v.y*n.x - v.x*n.y)*(v.y*n.x - v.x*n.y);
    current_AA[12] = (v.y*n.x - v.x*n.y)*n.x;
    current_AA[13] = (v.y*n.x - v.x*n.y)*n.y;
    current_AA[14] = (v.y*n.x - v.x*n.y)*n.z;
    
    current_AA[15] = n.x*n.x;
    current_AA[16] = n.x*n.y;
    current_AA[17] = n.x*n.z;
    
    current_AA[18] = n.y*n.y;
    current_AA[19] = n.y*n.z;
    
    current_AA[20] = n.z*n.z;
}

/// Transform Kinect depth measurements to milimeters.
struct transform_depth
{
    __host__ __device__
    float operator()(uint16_t a)
    {
        if(a == 2047)
            return 0.f;
        
        return 1000.f / (a * -0.0030711016f + 3.3309495161f);
    }
};

void Measurement::setDepth(uint16_t* depth)
{
    cudaSafeCall(cudaMemcpy(mRawDepthGpu, depth, sizeof(uint16_t)*mNumVertices,
            cudaMemcpyHostToDevice));
    thrust::transform(thrust::device_ptr<uint16_t>(mRawDepthGpu),
                      thrust::device_ptr<uint16_t>(mRawDepthGpu + mNumVertices),
                      thrust::device_ptr<float>(mDepthGpu),
                      transform_depth());
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaSafeCall(cudaBindTexture2D(0, &depth_texture, mDepthGpu, &channelDesc,
                    mWidth, mHeight, mWidth*sizeof(float)));
    
    depth_texture.normalized = false;
    depth_texture.filterMode = cudaFilterModePoint;
    depth_texture.addressMode[0] = cudaAddressModeClamp;
    depth_texture.addressMode[1] = cudaAddressModeClamp;
    
    cudaSafeCall(cudaMemcpyToSymbol(invK, mKdepth->getInverse(), sizeof(float)*9));
    
    float3* vertices;
    float3* normals;
    cudaGLMapBufferObject((void**)&vertices, mVertexBuffer);
    cudaGLMapBufferObject((void**)&normals, mNormalBuffer);
    dim3 grid, block(16,16,1);
    grid.x = (mWidth-1)/block.x + 1;
    grid.y = (mHeight-1)/block.y + 1;
    measure<<<grid,block>>>(vertices, normals, mMaskGpu,
                            //mKdepth->getInverseGpu(),
                            mWidth, mHeight, mWidth*12);
    cudaGLUnmapBufferObject(mVertexBuffer);
    cudaGLUnmapBufferObject(mNormalBuffer);
}

void VolumeFusion::initBoundingBox()
{
    float3 lower = gridToWorld(make_float3(0,0,0), mSide, mUnitsPerVoxel);
    float3 upper = gridToWorld(make_float3(mSide,mSide,mSide), mSide, mUnitsPerVoxel);
    mBoundingBox[0] = lower.x; mBoundingBox[1] = lower.y; mBoundingBox[2] = lower.z;
    mBoundingBox[3] = upper.x; mBoundingBox[4] = upper.y; mBoundingBox[5] = upper.z;
}

template<typename texture>
void VolumeFusion::bindTextureToF(texture& tex) const
{
    cudaSafeCall(cudaMemcpy3D(mCopyParams));
    
    tex.normalized = false;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    
    static const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaSafeCall(cudaBindTextureToArray(tex, mFArray, channelDesc));
}

void VolumeFusion::update(const Measurement& measurement, const float* T)
{
    dim3 block(8,8,8);
    dim3 grid;
    grid.x = grid.y = mSide/block.x;
    grid.z = 1;
    
    const float* kdepth = measurement.getKdepth()->get();
    const float* kdepthinv = measurement.getKdepth()->getInverse();
    cudaSafeCall(cudaMemcpyToSymbol(K, kdepth, sizeof(float)*9));
    cudaSafeCall(cudaMemcpyToSymbol(invK, kdepthinv, sizeof(float)*9));
    cudaSafeCall(cudaMemcpyToSymbol(Tgk, T, sizeof(float)*16));
    
    for(int i=0; i<mSide; i+=block.z)
        update_reconstruction<<<grid,block>>>(mFGpu, mWGpu, mSide, mUnitsPerVoxel, 200.f, i);
    cudaSafeCall(cudaGetLastError());
}

void VolumeMeasurement::measure(const VolumeFusion& volume, const float* T)
{
    volume.bindTextureToF(F_texture);
    
    float position[3];
    position[0] = T[3]; position[1] = T[7]; position[2] = T[11];
    float mindistance = volume.getMinimumDistanceTo(position);
    float maxdistance = volume.getMaximumDistanceTo(position);
    
    cudaSafeCall(cudaMemcpyToSymbol(invK, mKdepth->getInverse(), sizeof(float)*9));
    cudaSafeCall(cudaMemcpyToSymbol(Tgk, T, sizeof(float)*16));
    
    float3* vertices;
    float3* normals;
    cudaGLMapBufferObject((void**)&vertices, mVertexBuffer);
    cudaGLMapBufferObject((void**)&normals, mNormalBuffer);
    dim3 grid, block(16,16,1);
    grid.x = (mWidth-1)/block.x + 1;
    grid.y = (mHeight-1)/block.y + 1;
    raycast<<<grid,block>>>(vertices, normals, mWidth, mHeight, mWidth*12,
                            volume.getSide(), volume.getUnitsPerVoxel(), 200.f,
                            mindistance, maxdistance);
    cudaGLUnmapBufferObject(mVertexBuffer);
    cudaGLUnmapBufferObject(mNormalBuffer);
    
    // Preserve the current T.
    std::copy(T, T+16, mT);
}

void Tracker::trackStep(float* newT, const float* currentT, const float* current2InitT,
                        float3* verticesOld, float3* normalsOld,
                        float3* verticesNew, float3* normalsNew,
                        const Measurement& meas, const VolumeMeasurement& volMeas)
{
    thrust::fill(thrust::device_ptr<float>(mAAGpu),
                thrust::device_ptr<float>(mAAGpu + 21*mMaxNumVertices), 0.f);
    thrust::fill(thrust::device_ptr<float>(mAbGpu),
                thrust::device_ptr<float>(mAbGpu + 6*mMaxNumVertices), 0.f);
    ;
    dim3 block(16,16,1), grid;
    grid.x = (meas.getWidth() - 1)/block.x + 1;
    grid.y = (meas.getHeight() - 1)/block.y + 1;
    
    //cudaSafeCall(cudaMemcpy(mCurrentTGpu, currentT, sizeof(float)*16, cudaMemcpyHostToDevice));
    //cudaSafeCall(cudaMemcpy(mCurrent2InitTGpu, current2InitT, sizeof(float)*16, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpyToSymbol(Tgk, currentT, sizeof(float)*16));
    cudaSafeCall(cudaMemcpyToSymbol(Tk_1k, current2InitT, sizeof(float)*16));
    cudaSafeCall(cudaMemcpyToSymbol(K, meas.getKdepth()->get(), sizeof(float)*9));
    // __global__ void compute_tracking_matrices(float* AA, float* Ab,
    //                         float3* vertices_measure, float3* normals_measure,
    //                         float3* vertices_raycast, float3* normals_raycast,
    //                         int width, int height,
    //                         size_t AA_pitch, size_t Ab_pitch,
    //                         const int* mask,
    //                         float threshold_distance)
    compute_tracking_matrices<<<grid,block>>>(mAAGpu, mAbGpu,
                                              verticesOld, normalsOld,
                                              verticesNew, normalsNew,
                                              meas.getWidth(), meas.getHeight(),
                                              sizeof(float)*21, sizeof(float)*6,
                                              meas.getMaskGpu(),
                                              20.f);
}
