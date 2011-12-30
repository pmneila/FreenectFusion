
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

cuda_source = """

texture<float, cudaTextureType2D, cudaReadModeElementType> depth_texture;

__device__ __constant__ float K[9];
__device__ __constant__ float invK[9];

inline __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
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

__device__ float length(float3 v)
{
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline __device__ float3 normalize(float3 v)
{
    float invLen = 1.0f / length(v);
    return invLen * v;
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

__global__ void measure(float3* vertices, float3* normals,
                        int width, int height, size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x >= width || y >= height)
        return;
    
    float3* current_vertex = (float3*)((char*)vertices + pitch*y) + x;
    float3* current_normal = (float3*)((char*)normals + pitch*y) + x;
    
    float3 u = make_float3(float(x), float(y), 1.f);
    float3 v = make_float3(float(x+1), float(y), 1.f);
    float3 w = make_float3(float(x), float(y+1), 1.f);
    u = tex2D(depth_texture, x, y) * transform3(invK, u);
    v = tex2D(depth_texture, x+1, y) * transform3(invK, v);
    w = tex2D(depth_texture, x, y+1) * transform3(invK, w);
    
    float3 n = normalize(cross(v - u, w - u));
    *current_vertex = u;
    *current_normal = n;
}

__global__ void update_reconstruction(float* F, float* W,
                        int depth_width, int depth_height,
                        int side, float mu, int init_slice)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z + init_slice;
        
    float* F_data = F + k*side*side + j*side + i;
    float* W_data = W + k*side*side + j*side + i;
    
    // Point 3D.
    float3 p;
    p.x = (i - side/2) * 7.8125f;
    p.y = (j - side/2) * 7.8125f;
    p.z = (k + side/4) * 7.8125f;
    
    // Project the point.
    float3 x = transform3(K, p);
    x.x = round(x.x/x.z);
    x.y = round(x.y/x.z);
    x.z = 1.f;
    
    // Determine lambda.
    float3 aux = transform3(invK, x);
    float lambda = length(aux);
    
    float R = tex2D(depth_texture, x.x, x.y);
    
    lambda = R - length(p)/lambda;
    *F_data = min(1.f, lambda/mu);
    *W_data = 1.f;
    if(-lambda > mu || R == 0.f)
    {
        *F_data = NAN;
        *W_data = 0.f;
    }
}
"""

class FreenectFusion(object):
    
    def __init__(self, K_ir, K_rgb, T_rel, side=256, mu=200.0):
        self.K_ir = K_ir
        self.K_rgb = K_rgb
        self.T_rel = T_rel
        self.side = side
        self.mu = mu
        
        # Process the module.
        self.module = SourceModule(cuda_source)
        self.update_reconstruction = self.module.get_function("update_reconstruction")
        self.compute_depth = self.module.get_function("compute_depth")
        self.measure = self.module.get_function("measure")
        self.depth_texture = self.module.get_texref("depth_texture")
        print self.update_reconstruction.local_size_bytes
        print self.update_reconstruction.num_regs
        
        # Set the global constant matrices.
        mod = self.module # For short access.
        K, _ = mod.get_global("K")
        invK, _ = mod.get_global("invK")
        drv.memcpy_htod(K, np.float32(kc.K_ir.ravel()))
        drv.memcpy_htod(invK, np.float32(la.inv(kc.K_ir).ravel()))
        
        print drv.mem_get_info()
        self.F_gpu = gpuarray.zeros((side,)*3, dtype=np.float32)
        self.W_gpu = gpuarray.zeros((side,)*3, dtype=np.float32)
        self.vertices_gpu = gpuarray.empty((480,640), dtype=gpuarray.vec.float3)
        print drv.mem_get_info()
        print self.vertices_gpu.strides
    
    def update(self, depth, rgb_img=None):
        
        # Compute the real world depths.
        # TODO: Determine the best block size.
        depth_gpu = gpuarray.to_gpu(np.float32(depth))
        width = depth_gpu.shape[1]
        height = depth_gpu.shape[0]
        gridx = (width - 1) // 16 + 1
        gridy = (height - 1) // 16 + 1
        pitch = depth_gpu.strides[0]
        self.compute_depth(depth_gpu, np.int32(width), np.int32(height), np.int32(pitch),
                            block=(16,16,1), grid=(gridx, gridy))
        self.depth_gpu = depth_gpu
        
        # Prepare the depth array to be accessed as a texture.
        descr = drv.ArrayDescriptor()
        descr.width = width
        descr.height = height
        descr.format = drv.array_format.FLOAT
        descr.num_channels = 1
        self.depth_texture.set_address_2d(depth_gpu.gpudata, descr, pitch)
        
        # Extract vertices.
        #self.measure(self.vertices_gpu, self.vertices_gpu, np.int32(width), np.int32(height),
        #                        np.int32(pitch), block=(16,16,1), grid=(gridx, gridy))
        # Update the reconstruction.
        #gridx = int((self.side - 1) // 8 + 1)
        #for i in xrange(0, self.side, 8):
            #self.update_reconstruction(self.F_gpu, self.W_gpu, np.int32(width), np.int32(height),
                                    #np.int32(self.side), np.int32(self.mu), np.int32(i),
                                    #block=(8,8,8), grid=(gridx,gridx))
    

class DenseDemo(DemoBase):
    
    def __init__(self, width, height):
        super(DenseDemo, self).__init__(width, height)
        
        self.gl_rgb_texture = None
        self.gl_vertex_array = None
    
    def init_gl(self, width, height):
        super(DenseDemo, self).init_gl(width, height)
        
        import pycuda.gl.autoinit
        
        self.ffusion = FreenectFusion(kc.K_ir, kc.K_rgb, kc.T)
        freenect.sync_set_led(2)
        
        # Create a texture.
        self.gl_rgb_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_rgb_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        # Create a vertex buffer.
        self.gl_vertex_array = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.gl_vertex_array)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 640*480*24, None, gl.GL_DYNAMIC_COPY)
        self.gl_normal_array = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.gl_normal_array)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 640*480*24, None, gl.GL_DYNAMIC_COPY)
        # Register it with CUDA.
        self.vertex_buffer = cudagl.RegisteredBuffer(int(self.gl_vertex_array))
        self.normal_buffer = cudagl.RegisteredBuffer(int(self.gl_normal_array))
    
    def display(self):
        
        depth = freenect.sync_get_depth()[0]
        #depth = np.zeros((480,640), dtype=np.float32)
        self.ffusion.update(depth)
        
        vertex_map = self.vertex_buffer.map()
        normal_map = self.normal_buffer.map()
        ptr1, _ = vertex_map.device_ptr_and_size()
        ptr2, _ = normal_map.device_ptr_and_size()
        self.ffusion.measure(np.intp(ptr1), np.intp(ptr2),
                np.int32(640), np.int32(480), np.int32(7680),
                block=(16,16,1), grid=(40,30))
        vertex_map.unmap()
        normal_map.unmap()
        
        gl.glPointSize(1)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.gl_vertex_array)
        gl.glVertexPointer(3, gl.GL_FLOAT, 12, None)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.gl_normal_array)
        gl.glColorPointer(3, gl.GL_FLOAT, 12, ctypes.c_void_p(0))
        gl.glDrawArrays(gl.GL_POINTS, 0, 640*480)
        
        # Draw axes indicator.
        gl.glPointSize(5)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3d(1, 0, 0)
        gl.glVertex3d(100.0, 0, 0)
        gl.glColor3d(0, 1, 0)
        gl.glVertex3d(0, 100.0, 0)
        gl.glColor3d(0, 0, 1)
        gl.glVertex3d(0, 0, 100.0)
        gl.glEnd()
    
    def keyboard_press_event(self, key, x, y):
        if key == chr(27):
            freenect.sync_set_led(1)
            freenect.sync_stop()
            np.save("test", self.ffusion.vertices_gpu.get())
        
        super(DenseDemo, self).keyboard_press_event(key, x, y)
    

if __name__ == '__main__':
    DenseDemo(640, 480).run()
