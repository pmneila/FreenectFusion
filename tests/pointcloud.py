
from demobase import DemoBase

import numpy as np
import numpy.linalg as la
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import freenect

import kinect_calib as kc

class PointCloudDemo(DemoBase):
    
    def __init__(self, width, height):
        super(PointCloudDemo, self).__init__(width, height)
        
        self.gl_rgb_texture = None
        
        # Prepare the points.
        self.vertices = np.reshape(np.mgrid[:480,:640][[1,0]], (2,-1))
        self.vertices = np.insert(self.vertices, 2, 1, 0)
        
        # Prepare the transformation matrices.
        _aux = np.insert(np.insert(kc.K_rgb, 3, 0, 1), 2, 0, 0)
        self.texcoord_matrix = np.dot(_aux, kc.T).T
        self.vertices_matrix = np.vstack([np.insert(la.inv(kc.K_ir), 2, 0, 1),
                            [0,0,-0.0030711016e-3,3.3309495161e-3]]).T
        
        self.show_ir = False
    
    def init_gl(self, width, height):
        super(PointCloudDemo, self).init_gl(width, height)
        
        gl.glEnable(gl.GL_TEXTURE_2D)
        self.gl_rgb_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_rgb_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        freenect.sync_set_led(2)
    
    def display(self):
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_rgb_texture)
        
        # Prepare the texture matrix.
        gl.glMatrixMode(gl.GL_TEXTURE)
        gl.glLoadIdentity()
        gl.glScaled(1/640.0, 1/480.0, 1.0)
        
        # Get the Kinect image and prepare an OpenGL texture.
        if not self.show_ir:
            rgb_img = freenect.sync_get_video()[0]
            # Copy the image to the texture.
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, 3, 640, 480, 0,
                            gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_img)
            # Refine the texture matrix.
            gl.glMultMatrixd(self.texcoord_matrix)
            gl.glMultMatrixd(self.vertices_matrix)
        else:
            ir_img = freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)[0]
            # Copy the image to the texture.
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, 3, 640, 480, 0,
                            gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, ir_img)
        
        # Get the depth map.
        depth_img = freenect.sync_get_depth()[0]
        
        # Prepare the model-view matrix.
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glTranslated(0.0, 0.0, -1000.0)
        gl.glPushMatrix()
        gl.glMultMatrixd(self.vertices_matrix)
        
        # Prepare the vertices.
        self.vertices[2,:] = depth_img.ravel()
        
        # Draw the vertices.
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        data = self.vertices.T.ravel()
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
    
    def keyboard_press_event(self, key, x, y):
        
        if key == chr(27):
            freenect.sync_set_led(1)
            freenect.sync_stop()
        elif key == ' ':
            print "Toggled the RGB/IR image."
            self.show_ir = not self.show_ir
        
        super(PointCloudDemo, self).keyboard_press_event(key, x, y)
    

if __name__ == '__main__':
    PointCloudDemo(640, 480).run()
