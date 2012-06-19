
#include "FreenectFusion.h"

#include "cudautils.h"
#include "cudamath.h"
#include <cuda_gl_interop.h>
#include <thrust/transform.h>
#include <thrust/fill.h>

texture<float, 2, cudaReadModeElementType> depth_texture;
texture<float, 3, cudaReadModeElementType> F_texture;

__constant__ float K[9];
__constant__ float invK[9];
__constant__ float Tgk[16];
__constant__ float Tk_1k[16];

__device__ float3 transform3(const float* matrix, const float3& v)
{
    float3 res;
    res.x = matrix[0]*v.x + matrix[1]*v.y + matrix[2]*v.z;
    res.y = matrix[3]*v.x + matrix[4]*v.y + matrix[5]*v.z;
    res.z = matrix[6]*v.x + matrix[7]*v.y + matrix[8]*v.z;
    return res;
}

__device__ float3 transform3_affine(const float* matrix, const float3& v)
{
    float3 res;
    res.x = matrix[0]*v.x + matrix[1]*v.y + matrix[2]*v.z + matrix[3];
    res.y = matrix[4]*v.x + matrix[5]*v.y + matrix[6]*v.z + matrix[7];
    res.z = matrix[8]*v.x + matrix[9]*v.y + matrix[10]*v.z + matrix[11];
    return res;
}

__device__ float3 transform3_affine_inverse(const float* matrix, const float3& v)
{
    float3 res;
    float3 v2 = make_float3(v.x-matrix[3], v.y-matrix[7], v.z-matrix[11]);
    res.x = matrix[0]*v2.x + matrix[4]*v2.y + matrix[8]*v2.z;
    res.y = matrix[1]*v2.x + matrix[5]*v2.y + matrix[9]*v2.z;
    res.z = matrix[2]*v2.x + matrix[6]*v2.y + matrix[10]*v2.z;
    return res;
}

__device__ float gaussian(float t, float sigma)
{
    return exp(-t*t/(sigma*sigma));
}

__host__ __device__ float3 gridToWorld(const float3& p, int side, float units_per_voxel)
{
    return make_float3((p.x - side/2 + 0.5f) * units_per_voxel,
                        (p.y - side/2 + 0.5f) * units_per_voxel,
                        (p.z - side/2 + 0.5f) * units_per_voxel);
}

__host__ __device__ float3 worldToGrid(const float3& p, int side, float units_per_voxel)
{
    return make_float3(p.x/units_per_voxel + side/2 - 0.5f,
                        p.y/units_per_voxel + side/2 - 0.5f,
                        p.z/units_per_voxel + side/2 - 0.5f);
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
            float weight1 = gaussian(length2(make_float2(i,j)), sigma1);
            float weight2 = gaussian(depth1 - depth2, sigma2);
            weight_cum += weight1 * weight2;
            cum += depth2 * weight1 * weight2;
        }
    cum /= weight_cum;
    *current_smooth_depth = cum;
}

__global__ void pyrdownSmoothDepth(float* output, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x >= width || y >= height)
        return;
    
    float* current = &output[y*width + x];
    
    float depth1 = tex2D(depth_texture, 2*x, 2*y);
    float cum = 0.f;
    float weight_cum = 0.f;
    for(int i=-2; i<=2; ++i)
        for(int j=-2; j<=2; ++j)
        {
            float depth2 = tex2D(depth_texture, 2*x+i, 2*y+j);
            float weight1 = gaussian(length2(make_float2(i,j)), 1.f);
            float weight2 = gaussian(depth1 - depth2, 20.f);
            weight_cum += weight1 * weight2;
            cum += depth2 * weight1 * weight2;
        }
    cum /= weight_cum;
    *current = cum;
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
    if(depth < 0.01f)
    {
        *current_vertex = make_float3(0.f, 0.f, 0.f);
        *current_normal = make_float3(0.f, 0.f, 2.f);
    }
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
    *current_W = min(*current_W + W_rk, 50.f);
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
    float old_value = tex3D(F_texture, p.x+0.5f, p.y+0.5f, p.z+0.5f);
    for(float distance = mindistance; distance < maxdistance; distance += step)
    {
        p = worldToGrid(tgk + distance * ray, side, units_per_voxel);
        float value = tex3D(F_texture, p.x+0.5f, p.y+0.5f, p.z+0.5f);
        
        if(value < -2 || (old_value < 0 && value > 0))
            break;
        if(old_value >= 0 && value < 0)
        {
            float t = distance - step - (step * old_value)/(value - old_value);
            *current_vertex = tgk + t * ray;
            float valuex = tex3D(F_texture, p.x-1+0.5f, p.y+0.5f, p.z+0.5f);
            float valuey = tex3D(F_texture, p.x+0.5f, p.y-1+0.5f, p.z+0.5f);
            float valuez = tex3D(F_texture, p.x+0.5f, p.y+0.5f, p.z-1+0.5f);
            *current_normal = normalize(make_float3(valuex-value, valuey-value, valuez-value));
            return;
        }
        
        old_value = value;
    }
    *current_vertex = make_float3(0.f, 0.f, 0.f);
    *current_normal = make_float3(0.f, 0.f, 2.f);
}

__device__ float3 project(const float* K, const float* T, float3 point)
{
    return transform3(K, transform3_affine(T, point));
}

__device__ int2 hom2cart(float3 point)
{
    return make_int2(roundf(point.x/point.z), roundf(point.y/point.z));
}

__global__ void search_correspondences(float3* vertices_corresp, float3* normals_corresp,
            const float3* vertices_measure, float3* normals_measure,
            const float3* vertices_raycast, const float3* normals_raycast,
            int width_measure, int height_measure,
            int width_raycast, int height_raycast,
            float threshold_distance)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int thid = width_measure*y + x;
    
    if(x >= width_measure || y >= height_measure)
        return;
    
    float3* current_vertex_corresp = &vertices_corresp[thid];
    float3* current_normal_corresp = &normals_corresp[thid];
    
    float3 vertex_measure = vertices_measure[thid];
    
    // Get the corresponding pixel in the raycast image.
    int2 u_raycast = hom2cart(project(K, Tk_1k, vertex_measure));
    
    if(u_raycast.x < 0 || u_raycast.y < 0 ||
            u_raycast.x >= width_raycast || u_raycast.y >= height_raycast)
    {
        *current_vertex_corresp = make_float3(0.f, 0.f, 0.f);
        *current_normal_corresp = make_float3(0.f, 0.f, 2.f);
        return;
    }
    
    int id_raycast = width_raycast*u_raycast.y + u_raycast.x;
    
    float3 v = transform3_affine(Tgk, vertex_measure);
    float3 vdiff = vertices_raycast[id_raycast] - v;
    float vertex_distance = length(vdiff);
    
    // Prune invalid matches.
    if(vertex_measure.z==0.f || vertex_distance > threshold_distance)
    {
        *current_vertex_corresp = make_float3(0.f, 0.f, 0.f);
        *current_normal_corresp = make_float3(0.f, 0.f, 2.f);
        return;
    }
    
    *current_vertex_corresp = vertices_raycast[id_raycast];
    *current_normal_corresp = normals_raycast[id_raycast];
    
    // For debug only.
    normals_measure[thid] = make_float3(1.f, 1.f, 1.f);
}

__global__ void compute_tracking_matrices(float* AA, float* Ab,
                        const float3* vertices_measure, const float3* normals_measure,
                        const float3* vertices_corresp, const float3* normals_corresp,
                        int numVertices)
{
    int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    int thid = __mul24(blockId, blockDim.x) + threadIdx.x;
    //int thid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(thid >= numVertices)
        return;
    
    float* current_AA = &AA[21*thid];
    float* current_Ab = &Ab[6*thid];
    
    if(normals_corresp[thid].z == 2.f)
        return;
    
    float3 v = transform3_affine(Tgk, vertices_measure[thid]);
    float3 n = normals_corresp[thid];
    float b = dot(vertices_corresp[thid] - v, n);
    
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
    // Convert raw depth to milimeters.
    cudaSafeCall(cudaMemcpy(mRawDepthGpu, depth, sizeof(uint16_t)*mNumVertices,
            cudaMemcpyHostToDevice));
    thrust::transform(thrust::device_ptr<uint16_t>(mRawDepthGpu),
                      thrust::device_ptr<uint16_t>(mRawDepthGpu + mNumVertices),
                      thrust::device_ptr<float>(mDepthGpu),
                      transform_depth());
    
    // Generate the pyramid of depth maps and vertices/normals.
    for(int i=0; i<3; ++i)
        mPyramid[i]->update();
}

void PyramidMeasurement::update()
{
    const float* previousDepthGpu;
    int previousWidth, previousHeight;
    
    // Get info from the parent.
    if(mParent != 0)
    {
        previousDepthGpu = mParent->getDepthGpu();
        previousWidth = mParent->getWidth();
        previousHeight = mParent->getHeight();
    }
    else
    {
        previousDepthGpu = mParent2->getDepthGpu();
        previousWidth = mParent2->getWidth();
        previousHeight = mParent2->getHeight();
    }
    
    // Bind the depth into a texture for fast access.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaSafeCall(cudaBindTexture2D(0, &depth_texture, previousDepthGpu, &channelDesc,
                    previousWidth, previousHeight, previousWidth*sizeof(float)));
    depth_texture.normalized = false;
    depth_texture.filterMode = cudaFilterModePoint;
    depth_texture.addressMode[0] = cudaAddressModeBorder;
    depth_texture.addressMode[1] = cudaAddressModeBorder;
    
    // Smooth or resize the image down.
    dim3 grid, block(16,16,1);
    grid.x = (mWidth-1)/block.x + 1;
    grid.y = (mHeight-1)/block.y + 1;
    if(mParent != 0)
        compute_smooth_depth<<<grid,block>>>(mDepthGpu, previousWidth, previousHeight,
                                            mWidth*sizeof(float), 2.f, 20.f);
    else
        pyrdownSmoothDepth<<<grid,block>>>(mDepthGpu, mWidth, mHeight);
    
    // Bind the new reduced/smooth depth into a texture for fast access.
    cudaSafeCall(cudaBindTexture2D(0, &depth_texture, mDepthGpu, &channelDesc,
                    mWidth, mHeight, mWidth*sizeof(float)));
    
    // Determine vertices and normals from the depth map.
    cudaSafeCall(cudaMemcpyToSymbol(invK, mKInv, sizeof(float)*9));
    
    float3* vertices;
    float3* normals;
    cudaGLMapBufferObject((void**)&vertices, mVertexBuffer);
    cudaGLMapBufferObject((void**)&normals, mNormalBuffer);
    measure<<<grid,block>>>(vertices, normals, mMaskGpu,
                            mWidth, mHeight, mWidth*3*sizeof(float));
    cudaSafeCall(cudaGetLastError());
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
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.addressMode[2] = cudaAddressModeBorder;
    
    static const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaSafeCall(cudaBindTextureToArray(tex, mFArray, channelDesc));
}

void VolumeFusion::update(const Measurement& measurement, const float* T)
{
    dim3 block(8,8,8);
    dim3 grid;
    grid.x = grid.y = mSide/block.x;
    grid.z = 1;
    
    // Set instrinsic and extrinsics in constant memory.
    const float* kdepth = measurement.getK();
    const float* kdepthinv = measurement.getKInverse();
    cudaSafeCall(cudaMemcpyToSymbol(K, kdepth, sizeof(float)*9));
    cudaSafeCall(cudaMemcpyToSymbol(invK, kdepthinv, sizeof(float)*9));
    cudaSafeCall(cudaMemcpyToSymbol(Tgk, T, sizeof(float)*16));
    
    // Bind the depth map into the depth_texture.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaSafeCall(cudaBindTexture2D(0, &depth_texture, measurement.getDepthGpu(), &channelDesc,
                    measurement.getWidth(), measurement.getHeight(),
                    measurement.getWidth()*sizeof(float)));
    
    // Update the volume.
    for(int i=0; i<mSide; i+=block.z)
        update_reconstruction<<<grid,block>>>(mFGpu, mWGpu, mSide, mUnitsPerVoxel, 30.f, i);
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
    raycast<<<grid,block>>>(vertices, normals, mWidth, mHeight, mWidth*3*sizeof(float),
                            volume.getSide(), volume.getUnitsPerVoxel(), 30.f,
                            mindistance, maxdistance);
    cudaSafeCall(cudaGetLastError());
    cudaGLUnmapBufferObject(mVertexBuffer);
    cudaGLUnmapBufferObject(mNormalBuffer);
    
    // Preserve the current T.
    std::copy(T, T+16, mT);
}

template<int T>
struct floatN
{
    float a[T];
    
    __host__ __device__
    floatN()
    {
        for(int i=0; i<T; ++i)
            a[i] = 0.f;
    }
    
    __host__ __device__
    floatN(float v)
    {
        for(int i=0; i<T; ++i)
            a[i] = v;
    }
};

__device__
floatN<21> operator+(const floatN<21>& a, const floatN<21>& b)
{
    floatN<21> res;
    for(int i=0; i<21; ++i)
        res.a[i] = a.a[i] + b.a[i];
    return res;
}

__device__
floatN<6> operator+(const floatN<6>& a, const floatN<6>& b)
{
    floatN<6> res;
    for(int i=0; i<6; ++i)
        res.a[i] = a.a[i] + b.a[i];
    return res;
}

void Tracker::searchCorrespondences(float3* vertexCorresp, float3* normalsCorresp,
               const float* K_,
               const float* currentT, const float* current2InitT,
               const float3* verticesMeasure, const float3* normalsMeasure,
               const float3* verticesRaycast, const float3* normalsRaycast,
               int widthMeasure, int heightMeasure, int widthRaycast, int heightRaycast)
{
    // Copy intrinsic and extrinsic matrices to constant memory.
    cudaSafeCall(cudaMemcpyToSymbol(Tgk, currentT, sizeof(float)*16));
    cudaSafeCall(cudaMemcpyToSymbol(Tk_1k, current2InitT, sizeof(float)*16));
    cudaSafeCall(cudaMemcpyToSymbol(K, K_, sizeof(float)*9));
    
    dim3 block(16,16,1), grid;
    grid.x = (widthMeasure - 1)/block.x + 1;
    grid.y = (heightMeasure - 1)/block.y + 1;
    // Search the correspondences between device measurements and volume measurements.
    search_correspondences<<<grid,block>>>(vertexCorresp, normalsCorresp,
                    verticesMeasure, (float3*)normalsMeasure, verticesRaycast, normalsRaycast,
                    widthMeasure, heightMeasure, widthRaycast, heightRaycast, 100.f);
    cudaSafeCall(cudaGetLastError());
}

void Tracker::trackStep(float* AA, float* Ab, const float* currentT,
                        const float3* verticesMeasure, const float3* normalsMeasure,
                        const float3* verticesCorresp, const float3* normalsCorresp,
                        int numVertices)
{
    // Set the result matrices to 0.
    thrust::fill(thrust::device_ptr<float>(mAAGpu),
                thrust::device_ptr<float>(mAAGpu + 21*numVertices), 0.f);
    thrust::fill(thrust::device_ptr<float>(mAbGpu),
                thrust::device_ptr<float>(mAbGpu + 6*numVertices), 0.f);
    
    // Determine the grid and block sizes.
    dim3 block(32,1,1), grid;
    grid.x = (numVertices - 1)/block.x + 1;
    while(grid.x > 65535)
    {
        grid.x /= 2;
        grid.y *= 2;
    }
    // Copy the extrinsic matrix to the constant memory.
    cudaSafeCall(cudaMemcpyToSymbol(Tgk, currentT, sizeof(float)*16));
    
    // Compute the matrices.
    compute_tracking_matrices<<<grid,block>>>(mAAGpu, mAbGpu,
                                              verticesMeasure, normalsMeasure,
                                              verticesCorresp, normalsCorresp,
                                              numVertices);
    cudaSafeCall(cudaGetLastError());
    
    // Sum AA and Ab.
    floatN<21> _AA = thrust::reduce(thrust::device_ptr<floatN<21> >((floatN<21>*)mAAGpu),
                        thrust::device_ptr<floatN<21> >(((floatN<21>*)mAAGpu) + numVertices),
                        floatN<21>(0.f), thrust::plus<floatN<21> >());
    floatN<6> _Ab = thrust::reduce(thrust::device_ptr<floatN<6> >((floatN<6>*)mAbGpu),
                        thrust::device_ptr<floatN<6> >(((floatN<6>*)mAbGpu) + numVertices),
                        floatN<6>(0.f), thrust::plus<floatN<6> >());
    ;
    std::copy(_AA.a, _AA.a+21, AA);
    std::copy(_Ab.a, _Ab.a+6, Ab);
}
