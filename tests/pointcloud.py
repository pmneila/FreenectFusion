
import sys
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import freenect

window = None

def init_gl(width, height):
    gl.glClearColor(0.2, 0.2, 0.2, 0.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    resize_event(width, height)

def resize_event(width, height):
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    glut.glutSwapBuffers()

def keyboard_event(key, x, y):
    global window
    
    if key == chr(27):
        glut.glutDestroyWindow(window)
        sys.exit(0)

def mouse_button_event(button, state, x, y):
    pass

def mouse_moved_event(x, y):
    pass

def main():
    global window
    
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE
                                | glut.GLUT_ALPHA | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(640, 480)
    
    window = glut.glutCreateWindow("Point cloud test")
    
    glut.glutDisplayFunc(display)
    glut.glutIdleFunc(display)
    glut.glutReshapeFunc(resize_event)
    glut.glutKeyboardFunc(keyboard_event)
    glut.glutMouseFunc(mouse_button_event)
    glut.glutMotionFunc(mouse_moved_event)
    
    init_gl(640, 480)
    
    glut.glutMainLoop()

if __name__ == '__main__':
    main()
