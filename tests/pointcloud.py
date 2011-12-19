
import sys
from operator import sub

import numpy as np
import numpy.linalg as la
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import freenect

from orbitcamera import OrbitCamera
import kinect_calib as kc

class CameraState:
    NONE = 0
    ROTATION = 1
    TRANSLATION = 2
    ZOOM = 3

window = None

key_state = [False]*256
mouse_position = None

camera = OrbitCamera()
camera_state = CameraState.NONE

gl_rgb_texture = None
gl_depth_texture = None

# Prepare the points.
points = np.reshape(np.mgrid[:640,:480], (2,-1))
texcoords = points / np.array([[640.0], [480.0]])
points = np.insert(points, 2, 1, 0)
points = np.dot(la.inv(kc.K_rgb), points)*100

def init_gl(width, height):
    global gl_rgb_texture
    
    gl.glClearColor(0.2, 0.2, 0.2, 0.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl_rgb_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, gl_rgb_texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    
    resize_event(width, height)

def resize_event(width, height):
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    # glu.gluOrtho2D(0, width, 0, height)
    glu.gluPerspective(60, width/float(height), 50.0, 8000.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    
    rgb_img = freenect.sync_get_video()[0]
    
    gl.glBindTexture(gl.GL_TEXTURE_2D, gl_rgb_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, 3, 640, 480, 0,
                    gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_img)
    
    gl.glLoadIdentity()
    glu.gluLookAt(*camera.get_glulookat_parameters())
    gl.glBegin(gl.GL_QUADS)
    gl.glTexCoord2d(0.0, 1.0)
    gl.glVertex3d(-10, -10, -10)
    gl.glTexCoord2d(0.0, 0.0)
    gl.glVertex3d(-10, 10, -10)
    gl.glTexCoord2d(1.0, 0.0)
    gl.glVertex3d(10, 10, -10)
    gl.glTexCoord2d(1.0, 1.0)
    gl.glVertex3d(10, -10, -10)
    gl.glEnd()
    
    glut.glutSwapBuffers()

def keyboard_press_event(key, x, y):
    global window
    global key_state
    
    if key == chr(27):
        freenect.sync_set_led(1)
        glut.glutDestroyWindow(window)
        sys.exit(0)
    
    key_state[ord(key)] = True

def keyboard_up_event(key, x, y):
    global key_state
    key_state[ord(key)] = False

def mouse_button_event(button, state, x, y):
    
    global mouse_position, camera_state
    
    mouse_position = (x,y)
    
    if state == glut.GLUT_UP:
        camera_state = CameraState.NONE
    elif button == glut.GLUT_LEFT_BUTTON:
        camera_state = CameraState.ROTATION
    elif button == glut.GLUT_RIGHT_BUTTON:
        camera_state = CameraState.ZOOM
    elif button == glut.GLUT_MIDDLE_BUTTON:
        camera_state = CameraState.TRANSLATION

def mouse_moved_event(x, y):
    
    global mouse_position, camera, camera_state
    
    offset = map(sub, (x,y), mouse_position)
    
    if camera_state == CameraState.ROTATION:
        camera.rotate(*[i/100.0 for i in offset])
    elif camera_state == CameraState.TRANSLATION:
        camera.translate(offset[0], -offset[1])
    elif camera_state == CameraState.ZOOM:
        camera.zoom(offset[1])
    
    mouse_position = (x,y)

def main():
    global window
    
    freenect.sync_set_led(2)
    
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE
                                | glut.GLUT_ALPHA | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(640, 480)
    
    window = glut.glutCreateWindow("Point cloud test")
    
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
