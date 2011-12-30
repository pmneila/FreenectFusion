
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

texture<float, cudaTextureType1D, cudaReadModeElementType> depth_texture;

__device__ __constant__ float K[9];
__device__ __constant__ float invK[9];

__device__ float3 transform3(float* matrix, float3 v)
{
    float3 res;
    res.x = matrix[0]*v.x + matrix[1]*v.y + matrix[2]*v.z;
    res.y = matrix[3]*v.x + matrix[4]*v.y + matrix[5]*v.z;
    res.z = matrix[6]*v.x + matrix[7]*v.y + matrix[8]*v.z;
    return res;
}

__device__ float norm2(float3 v)
{
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
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

__global__ void update_reconstruction(float* F, float* W,
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
    p.z = (k + side/2) * 7.8125f;
    
    // Project the point.
    float3 x = transform3(K, p);
    x.x = round(x.x/x.z);
    x.y = round(x.y/x.z);
    x.z = 1.f;
    
    // Determine lambda.
    float3 aux = transform3(invK, x);
    float lambda = norm2(aux);
    
    *F_data = norm2(p)/lambda;
    *W_data = tex1Dfetch(depth_texture, int(x.y)*640 + int(x.x));
}

__global__ void cosa(float* vertices, int width)
{
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    int index = 8*(y*width + x);
    vertices[index] = x*100.f;
    vertices[index + 1] = y*100.f;
    vertices[index + 2] = 10.f;
    vertices[index + 3] = 1.f;
    vertices[index + 4] = x/10.f;
    vertices[index + 5] = y/10.f;
    vertices[index + 6] = 1.f;
    vertices[index + 7] = 1.f;
}
"""

class FreenectFusion(object):
    
    def __init__(self, K_ir, K_rgb, T_rel, side=256, mu=200.0):
        self.K_ir = K_ir
        self.K_rgb = K_rgb
        self.T_rel = T_rel
        self.side = np.int32(side)
        self.mu = np.float32(mu)
        
        # Process the module.
        self.module = SourceModule(cuda_source)
        self.cosa = self.module.get_function("cosa")
        self.update_reconstruction = self.module.get_function("update_reconstruction")
        self.compute_depth = self.module.get_function("compute_depth")
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
        print drv.mem_get_info()
    
    def update(self, depth, rgb_img=None):
        
        # Compute the real world depths.
        # TODO: Determine the best block size.
        depth_gpu = gpuarray.to_gpu(np.float32(depth))
        width = np.int32(depth_gpu.shape[1])
        height = np.int32(depth_gpu.shape[0])
        gridx = int((width - 1) // 16 + 1)
        gridy = int((height - 1) // 16 + 1)
        self.compute_depth(depth_gpu, width, height, np.int32(depth_gpu.strides[0]),
                            block=(16,16,1), grid=(gridx, gridy))
        self.depth_gpu = depth_gpu
        depth_gpu.bind_to_texref(self.depth_texture)
        
        # Update the reconstruction.
        gridx = int((self.side - 1) // 8 + 1)
        for i in xrange(0, self.side, 8):
            self.update_reconstruction(self.F_gpu, self.W_gpu, depth_gpu,
                                    self.side, self.mu, np.int32(i),
                                    block=(8,8,8), grid=(gridx,gridx))
    

class DenseDemo(DemoBase):
    
    def __init__(self, width, height):
        super(DenseDemo, self).__init__(width, height)
        
        self.gl_rgb_texture = None
        self.gl_vertex_array = None
    
    def init_gl(self, width, height):
        super(DenseDemo, self).init_gl(width, height)
        
        freenect.sync_set_led(2)
        
        import pycuda.gl.autoinit
        
        self.ffusion = FreenectFusion(kc.K_ir, kc.K_rgb, kc.T)
        
        # Create a texture.
        self.gl_rgb_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_rgb_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        # Create a vertex buffer.
        self.gl_vertex_array = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.gl_vertex_array)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 100*32, None, gl.GL_DYNAMIC_COPY)
        # Register it with CUDA.
        self.cuda_buffer = cudagl.RegisteredBuffer(int(self.gl_vertex_array))
        
        # Modify the buffer with CUDA.
        mapping = self.cuda_buffer.map()
        ptr, _ = mapping.device_ptr_and_size()
        self.ffusion.cosa(np.intp(ptr), np.int32(10),  block=(10,10,1), grid=(1,1))
        mapping.unmap()
    
    def display(self):
        
        depth = freenect.sync_get_depth()[0]
        #depth = np.zeros((480,640), dtype=np.float32)
        self.ffusion.update(depth)
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.gl_vertex_array)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 32, None)
        gl.glColorPointer(4, gl.GL_FLOAT, 32, ctypes.c_void_p(16))
        gl.glDrawArrays(gl.GL_POINTS, 0, 100)
        
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
            np.save("test", self.ffusion.W_gpu.get())
        
        super(DenseDemo, self).keyboard_press_event(key, x, y)
    

if __name__ == '__main__':
    DenseDemo(640, 480).run()
