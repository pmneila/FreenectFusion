
import sys
from operator import sub

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

from orbitcamera import OrbitCamera

class CameraState:
    NONE = 0
    ROTATION = 1
    TRANSLATION = 2
    ZOOM = 3

class DemoBase(object):
    
    def __init__(self, width, height):
        self.window = None
        self.window_width, self.window_height = width, height
        self.key_state = [False]*256
        self.mouse_position = None
        self.camera = OrbitCamera()
        self.camera_state = CameraState.NONE
    
    def run(self):
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE
                                    | glut.GLUT_ALPHA | glut.GLUT_DEPTH)
        glut.glutInitWindowSize(self.window_width, self.window_height)
        
        self.window = glut.glutCreateWindow("Point cloud test")
        
        glut.glutDisplayFunc(self.display_base)
        glut.glutIdleFunc(self.display_base)
        glut.glutReshapeFunc(self.resize_event)
        glut.glutKeyboardFunc(self.keyboard_press_event)
        glut.glutKeyboardUpFunc(self.keyboard_up_event)
        glut.glutMouseFunc(self.mouse_button_event)
        glut.glutMotionFunc(self.mouse_moved_event)
        
        self.init_gl(self.window_width, self.window_height)
        
        glut.glutMainLoop()
    
    def init_gl(self, width, height):
        gl.glClearColor(0.2, 0.2, 0.2, 0.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        self.resize_event(width, height)
    
    def resize_event(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(60, width/float(height), 50.0, 8000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
        self.window_width = width
        self.window_height = height
    
    def display_base(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Prepare the model-view matrix.
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluLookAt(*self.camera.get_glulookat_parameters())
        
        self.display()
        
        glut.glutSwapBuffers()
    
    def keyboard_press_event(self, key, x, y):
        
        if key == chr(27):
            glut.glutDestroyWindow(self.window)
            sys.exit(0)
        
        self.key_state[ord(key)] = True
    
    def keyboard_up_event(self, key, x, y):
        self.key_state[ord(key)] = False
    
    def mouse_button_event(self, button, state, x, y):
        self.mouse_position = (x,y)
        
        if state == glut.GLUT_UP:
            self.camera_state = CameraState.NONE
        elif button == glut.GLUT_LEFT_BUTTON:
            self.camera_state = CameraState.ROTATION
        elif button == glut.GLUT_RIGHT_BUTTON:
            self.camera_state = CameraState.ZOOM
        elif button == glut.GLUT_MIDDLE_BUTTON:
            self.camera_state = CameraState.TRANSLATION
    
    def mouse_moved_event(self, x, y):
        offset = map(sub, (x,y), self.mouse_position)
        
        if self.camera_state == CameraState.ROTATION:
            self.camera.rotate(*[i/100.0 for i in offset])
        elif self.camera_state == CameraState.TRANSLATION:
            self.camera.translate(offset[0]*3, -offset[1]*3)
        elif self.camera_state == CameraState.ZOOM:
            self.camera.zoom(offset[1])
        
        self.mouse_position = (x,y)
    
