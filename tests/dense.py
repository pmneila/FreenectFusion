
import sys

import numpy as np
import numpy.linalg as la

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import freenect

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gl as cudagl
from pycuda.compiler import SourceModule

from orbitcamera import OrbitCamera
import kinect_calib as kc

mod = SourceModule("""
__global__ void cosa()
{
}
""")
cosa = mod.get_function("cosa")

class FreenectFusion(object):
    
    def __init__(self, K_ir, K_rgb, T_rel):
        self.K_ir = K_ir
        self.K_rgb = K_rgb
        self.T_rel = T_rel
    
    def update(depth, rgb_img=None):
        pass
    

class CameraState:
    NONE = 0
    ROTATION = 1
    TRANSLATION = 2
    ZOOM = 3

class GlobalState:
    """Global state for the demo."""
    window = None
    
    key_state = [False]*256
    mouse_position = None
    
    camera = OrbitCamera()
    camera_state = CameraState.NONE
    
    gl_rgb_texture = None
    gl_vertex_array = None
    
    #depth_map = np.array([123.6 * np.tan(i/2842.5 + 1.1863) for i in xrange(2048)])
    #depth_map = np.array([1000.0 / (i * -0.0030711016 + 3.3309495161) for i in xrange(2048)])
    #depth_map[2047] = 0.0
    
    show_ir = False

glbstate = GlobalState()

def init_gl(width, height):
    gl.glClearColor(0.2, 0.2, 0.2, 0.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_TEXTURE_2D)
    glbstate.gl_rgb_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, glbstate.gl_rgb_texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    
    glbstate.gl_vertex_array = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, glbstate.gl_vertex_array)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, 200*16, NULL, gl.GL_DYNAMIC_COPY)
    
    
    resize_event(width, height)

def resize_event(width, height):
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(60, width/float(height), 50.0, 8000.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    glut.glutSwapBuffers()

def keyboard_press_event(key, x, y):
    if key == chr(27):
        freenect.sync_set_led(1)
        glut.glutDestroyWindow(glbstate.window)
        sys.exit(0)
    elif key == ' ':
        print "Toggled the RGB/IR image."
        glbstate.show_ir = not glbstate.show_ir
    
    glbstate.key_state[ord(key)] = True

def keyboard_up_event(key, x, y):
    glbstate.key_state[ord(key)] = False

def mouse_button_event(button, state, x, y):
    glbstate.mouse_position = (x,y)
    
    if state == glut.GLUT_UP:
        glbstate.camera_state = CameraState.NONE
    elif button == glut.GLUT_LEFT_BUTTON:
        glbstate.camera_state = CameraState.ROTATION
    elif button == glut.GLUT_RIGHT_BUTTON:
        glbstate.camera_state = CameraState.ZOOM
    elif button == glut.GLUT_MIDDLE_BUTTON:
        glbstate.camera_state = CameraState.TRANSLATION

def mouse_moved_event(x, y):
    offset = map(sub, (x,y), glbstate.mouse_position)
    
    if glbstate.camera_state == CameraState.ROTATION:
        glbstate.camera.rotate(*[i/100.0 for i in offset])
    elif glbstate.camera_state == CameraState.TRANSLATION:
        glbstate.camera.translate(offset[0]*3, -offset[1]*3)
    elif glbstate.camera_state == CameraState.ZOOM:
        glbstate.camera.zoom(offset[1])
    
    glbstate.mouse_position = (x,y)

def main():
    freenect.sync_set_led(2)
    
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE
                                | glut.GLUT_ALPHA | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(640, 480)
    
    glbstate.window = glut.glutCreateWindow("Point cloud test")
    
    glut.glutDisplayFunc(display)
    glut.glutIdleFunc(display)
    glut.glutReshapeFunc(resize_event)
    glut.glutKeyboardFunc(keyboard_press_event)
    glut.glutKeyboardUpFunc(keyboard_up_event)
    glut.glutMouseFunc(mouse_button_event)
    glut.glutMotionFunc(mouse_moved_event)
    
    init_gl(640, 480)
    
    glut.glutMainLoop()

if __name__ == '__main__':
    main()
