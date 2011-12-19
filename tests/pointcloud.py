
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
vertices = np.reshape(np.mgrid[:480,:640][[1,0]], (2,-1))
texcoords = vertices / np.array([[640.0], [480.0]])
vertices = np.insert(vertices, 2, 1, 0)
vertices = np.dot(la.inv(kc.K_ir), vertices)

#depth_map = np.array([123.6 * np.tan(i/2842.5 + 1.1863) for i in xrange(2048)])
depth_map = np.array([1000.0 / (i * -0.0030711016 + 3.3309495161) for i in xrange(2048)])
depth_map[2047] = 0.0

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
    glu.gluPerspective(60, width/float(height), 50.0, 8000.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    
    #rgb_img = freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)[0]
    rgb_img = freenect.sync_get_video()[0]
    depth_img = freenect.sync_get_depth()[0]
    depth = depth_map[depth_img]
    
    current_vertices = vertices * depth.ravel()
    
    # Extract texcoords.
    aux = np.insert(current_vertices, 3, 1, 0)
    aux = np.dot(kc.K_rgb, np.dot(kc.T, aux)[:3,:])
    aux = aux / aux[2,:]
    texcoords = aux[:2,:] / np.array([[640.0], [480.0]])
    
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, gl_rgb_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, 3, 640, 480, 0,
                    gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_img)
    
    gl.glLoadIdentity()
    glu.gluLookAt(*camera.get_glulookat_parameters())
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
    gl.glVertexPointer(3, gl.GL_DOUBLE, 0, current_vertices.T.ravel())
    gl.glTexCoordPointer(2, gl.GL_DOUBLE, 0, texcoords.T.ravel())
    gl.glPointSize(1)
    gl.glColor3d(1,1,1)
    gl.glDrawArrays(gl.GL_POINTS, 0, 640*480)
    
    # Draw axes indicator.
    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
    gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
    gl.glDisable(gl.GL_TEXTURE_2D)
    gl.glPointSize(5)
    gl.glBegin(gl.GL_POINTS)
    gl.glColor3d(1, 0, 0)
    gl.glVertex3d(100.0, 0, 0)
    gl.glColor3d(0, 1, 0)
    gl.glVertex3d(0, 100.0, 0)
    gl.glColor3d(0, 0, 1)
    gl.glVertex3d(0, 0, 100.0)
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
