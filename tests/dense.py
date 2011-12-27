
from demobase import DemoBase

import ctypes

import numpy as np
import numpy.linalg as la

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import freenect

#import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gl as cudagl
from pycuda.compiler import SourceModule

import kinect_calib as kc

cuda_source = """

__device__ __constant__ float K[9];
__device__ __constant__ float invK[9];

__global__ void update_reconstruction(float* F, float* W, float* depth)
{
    
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
#cosa = mod.get_function("cosa")

class FreenectFusion(object):
    
    def __init__(self, K_ir, K_rgb, T_rel):
        self.K_ir = K_ir
        self.K_rgb = K_rgb
        self.T_rel = T_rel
    
    def update(depth, rgb_img=None):
        pass
    

class DenseDemo(DemoBase):
    
    def __init__(self, width, height):
        super(DenseDemo, self).__init__(width, height)
        
        self.gl_rgb_texture = None
        self.gl_vertex_array = None
    
    def init_gl(self, width, height):
        super(DenseDemo, self).init_gl(width, height)
        
        import pycuda.gl.autoinit
        
        mod = SourceModule(cuda_source)
        cosa = mod.get_function("cosa")
        
        # Set the global constant matrices.
        K, _ = mod.get_global("K")
        invK, _ = mod.get_global("invK")
        drv.memcpy_htod(K, np.float32(kc.K_ir.ravel()))
        drv.memcpy_htod(invK, np.float32(la.inv(kc.K_ir).ravel()))
        
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
        cosa(np.intp(ptr), np.int32(10),  block=(10,10,1), grid=(1,1))
        mapping.unmap()
    
    def display(self):
        
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
            #freenect.sync_set_led(1)
            #freenect.sync_stop()
            pass
        
        super(DenseDemo, self).keyboard_press_event(key, x, y)
    

if __name__ == '__main__':
    DenseDemo(640, 480).run()
