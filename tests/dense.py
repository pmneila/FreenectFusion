
from operator import methodcaller
from itertools import product

from demobase import DemoBase

import ctypes

import numpy as np
import numpy.linalg as la

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import freenect

import pycuda.driver as drv
import pycuda.gl as cudagl
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import kinect_calib as kc

import cudareduce

cuda_source = """

texture<float, 2, cudaReadModeElementType> depth_texture;
texture<float, 2, cudaReadModeElementType> smooth_depth_texture;
texture<float, 3, cudaReadModeElementType> F_texture;

__device__ __constant__ float K[9];
__device__ __constant__ float invK[9];

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

__device__ float3 transform3(float* matrix, float3 v)
{
    float3 res;
    res.x = matrix[0]*v.x + matrix[1]*v.y + matrix[2]*v.z;
    res.y = matrix[3]*v.x + matrix[4]*v.y + matrix[5]*v.z;
    res.z = matrix[6]*v.x + matrix[7]*v.y + matrix[8]*v.z;
    return res;
}

__device__ float3 transform3_affine(float* matrix, float3 v)
{
    float3 res;
    res.x = matrix[0]*v.x + matrix[1]*v.y + matrix[2]*v.z + matrix[3];
    res.y = matrix[4]*v.x + matrix[5]*v.y + matrix[6]*v.z + matrix[7];
    res.z = matrix[8]*v.x + matrix[9]*v.y + matrix[10]*v.z + matrix[11];
    return res;
}

__device__ float3 transform3_affine_inverse(float* matrix, float3 v)
{
    float3 res;
    float3 v2 = make_float3(v.x-matrix[3], v.y-matrix[7], v.z-matrix[11]);
    res.x = matrix[0]*v2.x + matrix[4]*v2.y + matrix[8]*v2.z;
    res.y = matrix[1]*v2.x + matrix[5]*v2.y + matrix[9]*v2.z;
    res.z = matrix[2]*v2.x + matrix[6]*v2.y + matrix[10]*v2.z;
    return res;
}

inline __device__ float length(float3 v)
{
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
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

__device__ float3 gridToWorld(float3 p, int side, float units_per_voxel)
{
    return make_float3((p.x - side/2) * units_per_voxel,
                        (p.y - side/2) * units_per_voxel,
                        (p.z - side/2) * units_per_voxel);
}

__device__ float3 worldToGrid(float3 p, int side, float units_per_voxel)
{
    return make_float3(p.x/units_per_voxel + side/2,
                        p.y/units_per_voxel + side/2,
                        p.z/units_per_voxel + side/2);
}

__global__ void compute_depth(float* depth, int width, int height, size_t pitch)
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
__global__ void measure(float3* vertices, float3* normals, float* mask,
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

__global__ void update_reconstruction(float* F, float* W, float3* normals,
                        size_t normals_pitch, int side, float units_per_voxel,
                        float mu, int init_slice, float* Tgk)
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
                        int side, float units_per_voxel, float mu, float* Tgk,
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

__device__ float3 project(float* K, float* T, float3 point)
{
    return transform3(K, transform3_affine(T, point));
}

__device__ int2 hom2cart(float3 point)
{
    return make_int2(roundf(point.x/point.z), roundf(point.y/point.z));
}

__global__ void compute_tracking_matrices(float* AA, float* Ab, float* omega,
                        float3* vertices_measure, float3* normals_measure,
                        float3* vertices_raycast, float3* normals_raycast,
                        int width, int height,
                        size_t AA_pitch, size_t Ab_pitch,
                        float* mask, float* Tgk, float* Tgk1_k,
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
    int2 u_raycast = hom2cart(project(K, Tgk1_k, vertex_measure));
    
    if(u_raycast.x < 0 || u_raycast.y < 0 ||
        u_raycast.x >= width || u_raycast.y >= height)
    {
        omega[thid] = 0.f;
        return;
    }
    
    int id_raycast = width*u_raycast.y + u_raycast.x;
    float3 vertex_raycast = vertices_raycast[id_raycast];
    
    float3 v = transform3_affine(Tgk, vertex_measure);
    float3 vdiff = vertex_raycast - v;
    float vertex_distance = length(vdiff);
    
    // Prune invalid matches.
    if(mask[thid] < 0.5f || vertex_distance > threshold_distance)
    {
        omega[thid] = 0.f;
        return;
    }
    
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
    omega[thid] = 1.f;
}
"""

def distance_to_bbox(bbox, point):
    p1, p2 = bbox
    size = np.abs(p1 - p2)/2.0
    center = (p1 + p2)/2.0
    point = point - center
    point_bbox = np.copy(point)
    out_upper = point_bbox > size
    point_bbox[out_upper] = size[out_upper]
    out_lower = point_bbox < -size
    point_bbox[out_lower] = -size[out_lower]
    return np.linalg.norm(point_bbox - point)

def distance_farthest_to_bbox(bbox, point):
    bbox = np.asarray(bbox)
    corners = np.asarray(map(lambda x: bbox[x,[0,1,2]], product(*[[0,1]]*3)))
    distances = np.sqrt(((corners - point)**2).sum(1))
    return np.max(distances)

class FreenectFusion(object):
    
    def __init__(self, K_ir, K_rgb, T_rel, side=256, units_per_voxel=7.8125, mu=200.0):
        self.K_ir = K_ir
        self.K_rgb = K_rgb
        self.T_rel = T_rel
        self.side = side
        self.units_per_voxel = units_per_voxel
        self.mu = mu
        
        # Process the module.
        self.module = SourceModule(cuda_source)
        self.update_reconstruction = self.module.get_function("update_reconstruction")
        self.compute_depth = self.module.get_function("compute_depth")
        self.compute_smooth_depth = self.module.get_function("compute_smooth_depth")
        self.compute_tracking_matrices = self.module.get_function("compute_tracking_matrices")
        self.measure = self.module.get_function("measure")
        self.raycast = self.module.get_function("raycast")
        
        self.depth_texture = self.module.get_texref("depth_texture")
        self.smooth_depth_texture = self.module.get_texref("smooth_depth_texture")
        self.F_texture = self.module.get_texref("F_texture")
        self.depth_texture.set_filter_mode(drv.filter_mode.POINT)
        self.smooth_depth_texture.set_filter_mode(drv.filter_mode.POINT)
        self.F_texture.set_filter_mode(drv.filter_mode.LINEAR)
        #print self.update_reconstruction.shared_size_bytes
        #print self.update_reconstruction.num_regs
        
        # Set the global constant matrices.
        mod = self.module # For short access.
        K, _ = mod.get_global("K")
        invK, _ = mod.get_global("invK")
        drv.memcpy_htod(K, np.float32(kc.K_ir.ravel()))
        drv.memcpy_htod(invK, np.float32(la.inv(kc.K_ir).ravel()))
        
        # Reserve GPU variables.
        print drv.mem_get_info()
        self.buffers = {}
        self.T_gk = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-1000],[0,0,0,1]], dtype=np.float32)
        self.T_gk_gpu = gpuarray.to_gpu(self.T_gk[:3])
        self.Tgk1_k_gpu = gpuarray.to_gpu(np.eye(4, dtype=np.float32)[:3])
        
        self.F_gpu = gpuarray.zeros((side,)*3, dtype=np.float32) - 1000
        self.W_gpu = gpuarray.zeros((side,)*3, dtype=np.float32)
        self.mask_gpu = gpuarray.zeros(480*640, dtype=np.float32)
        self.smooth_depth_gpu = gpuarray.zeros((480,640), dtype=np.float32)
        
        for buffer_type in ['measure', 'raycast']:
            buffers = gl.glGenBuffers(2)
            for b in buffers:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, 640*480*12, None, gl.GL_DYNAMIC_COPY)
            buffers = map(lambda x: cudagl.RegisteredBuffer(int(x)), buffers)
            self.buffers[buffer_type] = buffers
        
        # Tracking data.
        self.AA_gpu = gpuarray.zeros((640*480, 21), dtype=np.float32)
        self.Ab_gpu = gpuarray.zeros((640*480, 6), dtype=np.float32)
        self.omega_gpu = gpuarray.zeros(640*480, dtype=np.float32)
        self.AA = np.empty(21, dtype=np.float32)
        self.Ab = np.empty(6, dtype=np.float32)
        
        self.active_tracking = False
        
        print drv.mem_get_info()
        
        self._prepare_F_texture()
    
    def _prepare_F_texture(self):
        
        descr = drv.ArrayDescriptor3D()
        descr.width = self.side
        descr.height = self.side
        descr.depth = self.side
        descr.format = drv.dtype_to_array_format(self.F_gpu.dtype)
        descr.num_channels = 1
        descr.flags = 0
        
        F_array = drv.Array(descr)
        
        copy = drv.Memcpy3D()
        copy.set_src_device(self.F_gpu.gpudata)
        copy.set_dst_array(F_array)
        copy.width_in_bytes = copy.src_pitch = self.F_gpu.strides[1]
        copy.src_height = copy.height = self.side
        copy.depth = self.side
        
        self.F_gpu_to_array_copy = copy
        self.F_gpu_to_array_copy()
        self.F_texture.set_array(F_array)
    
    def grid_to_world(self, p):
        return (np.asarray(p) - self.side/2) * self.units_per_voxel
    
    def get_bounding_box(self):
        
        p1 = self.grid_to_world([0]*3)
        p2 = self.grid_to_world([self.side]*3)
        return p1, p2
    
    def get_gl_buffers(self, buffer_type='raycast'):
        return map(methodcaller("gl_handle"), self.buffers[buffer_type])
    
    def update(self, depth, rgb_img=None):
        
        # Compute the real world depths.
        # TODO: Determine the best block size.
        depth_gpu = gpuarray.to_gpu(np.float32(depth))
        width = depth_gpu.shape[1]
        height = depth_gpu.shape[0]
        gridx = (width - 1) // 16 + 1
        gridy = (height - 1) // 16 + 1
        pitch = depth_gpu.strides[0]
        self.compute_depth(depth_gpu, np.int32(width), np.int32(height), np.intp(pitch),
                            block=(16,16,1), grid=(gridx, gridy))
        
        # Prepare the depth array to be accessed as a texture.
        descr = drv.ArrayDescriptor()
        descr.width = width
        descr.height = height
        descr.format = drv.array_format.FLOAT
        descr.num_channels = 1
        self.depth_texture.set_address_2d(depth_gpu.gpudata, descr, pitch)
        
        # Smooth depth.
        # pitch = self.smooth_depth_gpu.strides[0]
        # self.compute_smooth_depth(self.smooth_depth_gpu, np.int32(width), np.int32(height),
        #                     np.intp(pitch), np.float32(10.0), np.float32(10000.0),
        #                     block=(16,16,1), grid=(gridx, gridy))
        #self.smooth_depth_texture.set_address_2d(depth_gpu.gpudata, descr, pitch)
        
        # Buffer mapping.
        normals_pitch = 640*12
        vertex_measure_map, normal_measure_map = map(methodcaller("map"), self.buffers["measure"])
        vertices_measure = np.intp(vertex_measure_map.device_ptr_and_size()[0])
        normals_measure = np.intp(normal_measure_map.device_ptr_and_size()[0])
        vertex_raycast_map, normal_raycast_map = map(methodcaller("map"), self.buffers["raycast"])
        vertices_raycast = np.intp(vertex_raycast_map.device_ptr_and_size()[0])
        normals_raycast = np.intp(normal_raycast_map.device_ptr_and_size()[0])
        
        # Measure
        self.measure(vertices_measure, normals_measure, self.mask_gpu, np.int32(width), np.int32(height),
                                np.intp(normals_pitch),
                                block=(16,16,1), grid=(gridx, gridy))
        
        # Update the reconstruction.
        grid2 = int((self.side - 1) // 8 + 1)
        for i in xrange(0, self.side, 8):
            self.update_reconstruction(self.F_gpu, self.W_gpu, normals_measure, np.intp(normals_pitch),
                                np.int32(self.side), np.float32(self.units_per_voxel),
                                np.float32(self.mu), np.int32(i),
                                self.T_gk_gpu,
                                block=(8,8,8), grid=(grid2,grid2))
        
        # Copy F from gpu to F_array (binded to F_texture).
        self.F_gpu_to_array_copy()
        
        # Raycast.
        bbox = self.get_bounding_box()
        point = self.T_gk[:3,3]
        mindistance = distance_to_bbox(bbox, point)
        maxdistance = distance_farthest_to_bbox(bbox, point)
        self.raycast(vertices_raycast, normals_raycast, np.int32(width), np.int32(height),
                                np.intp(normals_pitch),
                                np.int32(self.side), np.float32(self.units_per_voxel),
                                np.float32(self.mu), self.T_gk_gpu,
                                np.float32(mindistance), np.float32(maxdistance),
                                block=(16,16,1), grid=(gridx,gridy))
        
        # Tracking.
        # __global__ void compute_tracking_matrices(float* AA, float* Ab, float* omega,
        #                         float3* vertices_measure, float3* normals_measure,
        #                         float3* vertices_raycast, float3* normals_raycast,
        #                         int width, int height, size_t A_pitch,
        #                         float* mask, float* Tgk, float* Tgk1_k,
        #                         float threshold_distance)
        if self.active_tracking:
            self.AA_gpu.fill(0)
            self.Ab_gpu.fill(0)
            self.compute_tracking_matrices(self.AA_gpu, self.Ab_gpu, self.omega_gpu,
                                    vertices_measure, normals_measure,
                                    vertices_raycast, normals_raycast,
                                    np.int32(width), np.int32(height),
                                    np.intp(self.AA_gpu.strides[0]), np.intp(self.Ab_gpu.strides[0]),
                                    self.mask_gpu, self.T_gk_gpu, self.Tgk1_k_gpu, np.float32(20.0),
                                    block=(16,16,1), grid=(gridx,gridy))
            
            cudareduce.add_vectors(self.AA_gpu, 640*480, 21)
            cudareduce.add_vectors(self.Ab_gpu, 640*480, 6)
            drv.memcpy_dtoh(self.AA, self.AA_gpu.gpudata)
            drv.memcpy_dtoh(self.Ab, self.Ab_gpu.gpudata)
            
            # Solve the system.
            AA = np.zeros((6,6))
            AA[np.triu_indices(6)] = self.AA
            AA.T[np.triu_indices(6)] = self.AA
            try:
                x = np.linalg.solve(AA, self.Ab)
                Tinc = np.array([[1, x[2], -x[1], x[3]],
                                [-x[2], 1, x[0], x[4]],
                                [x[1], -x[0], 1, x[5]],
                                [0, 0, 0, 1]])
                U,D,V = np.linalg.svd(Tinc[:3,:3])
                Tinc[:3,:3] = np.dot(U, V)
            except np.linalg.LinAlgError:
                Tinc = np.eye(4)
            
            self.T_gk = np.float32(np.dot(Tinc, self.T_gk))
            self.T_gk_gpu = gpuarray.to_gpu(self.T_gk[:3])
        
        vertex_raycast_map.unmap()
        normal_raycast_map.unmap()
        vertex_measure_map.unmap()
        normal_measure_map.unmap()
    

class DenseDemo(DemoBase):
    
    def __init__(self, width, height):
        super(DenseDemo, self).__init__(width, height)
        
        self.gl_rgb_texture = None
        self.gl_vertex_array = None
        
        self.draw_flags = {'measure': False, 'raycast': True}
    
    def init_gl(self, width, height):
        super(DenseDemo, self).init_gl(width, height)
        
        import pycuda.gl.autoinit
        print "CUDA version: %s" % str(drv.get_version())
        print "CUDA driver version: %s" % drv.get_driver_version()
        print "CUDA device: %s" % pycuda.gl.autoinit.device.name()
        print "\tCompute capability: %s" % str(pycuda.gl.autoinit.device.compute_capability())
        print "\tTotal memory: %s" % pycuda.gl.autoinit.device.total_memory()
        
        self.ffusion = FreenectFusion(kc.K_ir, kc.K_rgb, kc.T, side=128)
        self.bbox = self.ffusion.get_bounding_box()
        #freenect.sync_set_led(2)
        
        # Create a texture.
        self.gl_rgb_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_rgb_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    
    def display(self):
        
        depth = freenect.sync_get_depth()[0]
        self.ffusion.update(depth)
        
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glRotatef(180, 0, 0, 1)
        
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        
        # Plot measure.
        if self.draw_flags['measure']:
            gl.glPushMatrix()
            gl.glMultMatrixf(self.ffusion.T_gk.T)
            gl.glPointSize(1)
            vertices, normals = self.ffusion.get_gl_buffers('measure')
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertices)
            gl.glVertexPointer(3, gl.GL_FLOAT, 12, None)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, normals)
            gl.glColorPointer(3, gl.GL_FLOAT, 12, ctypes.c_void_p(0))
            gl.glDrawArrays(gl.GL_POINTS, 0, 640*480)
            gl.glPopMatrix()
        
        # Plot raycast.
        if self.draw_flags['raycast']:
            gl.glPointSize(1)
            vertices, normals = self.ffusion.get_gl_buffers('raycast')
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertices)
            gl.glVertexPointer(3, gl.GL_FLOAT, 12, None)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, normals)
            gl.glColorPointer(3, gl.GL_FLOAT, 12, ctypes.c_void_p(0))
            gl.glDrawArrays(gl.GL_POINTS, 0, 640*480)
        
        # Draw axes indicator.
        gl.glPushMatrix()
        gl.glMultMatrixf(self.ffusion.T_gk.T)
        gl.glPointSize(5)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3d(1, 1, 1)
        gl.glVertex3d(0, 0, 0)
        gl.glColor3d(1, 0, 0)
        gl.glVertex3d(100.0, 0, 0)
        gl.glColor3d(0, 1, 0)
        gl.glVertex3d(0, 100.0, 0)
        gl.glColor3d(0, 0, 1)
        gl.glVertex3d(0, 0, 100.0)
        gl.glEnd()
        gl.glPopMatrix()
        
        # Draw bounding box.
        self.draw_bounding_box()
    
    def draw_bounding_box(self):
        p1, p2 = self.bbox
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glColor3d(1, 1, 1)
        gl.glVertex3d(p1[0], p1[1], p1[2])
        gl.glVertex3d(p2[0], p1[1], p1[2])
        gl.glVertex3d(p2[0], p2[1], p1[2])
        gl.glVertex3d(p1[0], p2[1], p1[2])
        gl.glEnd()
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glColor3d(1, 1, 1)
        gl.glVertex3d(p1[0], p1[1], p2[2])
        gl.glVertex3d(p2[0], p1[1], p2[2])
        gl.glVertex3d(p2[0], p2[1], p2[2])
        gl.glVertex3d(p1[0], p2[1], p2[2])
        gl.glEnd()
        gl.glBegin(gl.GL_LINES)
        gl.glColor3d(1, 1, 1)
        gl.glVertex3d(p1[0], p1[1], p1[2])
        gl.glVertex3d(p1[0], p1[1], p2[2])
        gl.glVertex3d(p2[0], p1[1], p1[2])
        gl.glVertex3d(p2[0], p1[1], p2[2])
        gl.glVertex3d(p2[0], p2[1], p1[2])
        gl.glVertex3d(p2[0], p2[1], p2[2])
        gl.glVertex3d(p1[0], p2[1], p1[2])
        gl.glVertex3d(p1[0], p2[1], p2[2])
        gl.glEnd()
    
    def keyboard_press_event(self, key, x, y):
        if key == chr(27):
            #freenect.sync_set_led(1)
            freenect.sync_stop()
            #np.savez("data", vertices=self.vertices_measure.get())
            #np.save("F", self.ffusion.F_gpu.get())
            #np.save("W", self.ffusion.W_gpu.get())
        if key == 'm':
            self.draw_flags['measure'] ^= True
        if key == 'r':
            self.draw_flags['raycast'] ^= True
        if key == 't':
            self.ffusion.active_tracking ^= True
        
        super(DenseDemo, self).keyboard_press_event(key, x, y)
    

if __name__ == '__main__':
    DenseDemo(640, 480).run()
