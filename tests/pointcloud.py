
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

class GlobalState:
    """Global state for the demo."""
    window = None
    
    key_state = [False]*256
    mouse_position = None
    
    camera = OrbitCamera()
    camera_state = CameraState.NONE
    
    gl_rgb_texture = None
    
    # Prepare the points.
    vertices = np.reshape(np.mgrid[:480,:640][[1,0]], (2,-1))
    vertices = np.insert(vertices, 2, 1, 0)
    
    #depth_map = np.array([123.6 * np.tan(i/2842.5 + 1.1863) for i in xrange(2048)])
    #depth_map = np.array([1000.0 / (i * -0.0030711016 + 3.3309495161) for i in xrange(2048)])
    #depth_map[2047] = 0.0
    
    # Prepare the transformation matrices.
    _aux = np.insert(np.insert(kc.K_rgb, 3, 0, 1), 2, 0, 0)
    texcoord_matrix = np.dot(_aux, kc.T).T
    vertices_matrix = np.vstack([np.insert(la.inv(kc.K_ir), 2, 0, 1),
                        [0,0,-0.0030711016e-3,3.3309495161e-3]]).T
    
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
    
    resize_event(width, height)

def resize_event(width, height):
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(60, width/float(height), 50.0, 8000.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, glbstate.gl_rgb_texture)
    
    # Prepare the texture matrix.
    gl.glMatrixMode(gl.GL_TEXTURE)
    gl.glLoadIdentity()
    gl.glScaled(1/640.0, 1/480.0, 1.0)
    
    # Get the Kinect image and prepare an OpenGL texture.
    if not glbstate.show_ir:
        rgb_img = freenect.sync_get_video()[0]
        # Copy the image to the texture.
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, 3, 640, 480, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_img)
        # Refine the texture matrix.
        gl.glMultMatrixd(glbstate.texcoord_matrix)
        gl.glMultMatrixd(glbstate.vertices_matrix)
    else:
        ir_img = freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)[0]
        # Copy the image to the texture.
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, 3, 640, 480, 0,
                        gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, ir_img)
        
    # Get the depth map.
    depth_img = freenect.sync_get_depth()[0]
    
    # Prepare the model-view matrix.
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glu.gluLookAt(*glbstate.camera.get_glulookat_parameters())
    gl.glTranslated(0.0, 0.0, -1000.0)
    gl.glPushMatrix()
    gl.glMultMatrixd(glbstate.vertices_matrix)
    
    # Prepare the vertices.
    glbstate.vertices[2,:] = depth_img.ravel()
    
    # Draw the vertices.
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
    data = glbstate.vertices.T.ravel()
    gl.glVertexPointer(3, gl.GL_DOUBLE, 0, data)
    gl.glTexCoordPointer(3, gl.GL_DOUBLE, 0, data)
    gl.glPointSize(2.0)
    gl.glColor3d(1,1,1)
    gl.glDrawArrays(gl.GL_POINTS, 0, 640*480)
    
    gl.glPopMatrix()
    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
    gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
    gl.glDisable(gl.GL_TEXTURE_2D)
    
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
