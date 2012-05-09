
#include "DemoBase.h"
#include "FreenectFusion.h"

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include <libfreenect_sync.h>

class Viewer : public DemoBase
{
private:
    GLuint mTexture;
    FreenectFusion mFfusion;
    
public:
    Viewer(int width, int height)
        : DemoBase(width, height),
        mFfusion(640, 480, 0, 0)
    {}
    
protected:
    void display()
    {
        void* image = 0;
        void* depth = 0;
        uint32_t timestamp;
        freenect_sync_get_video(&image, &timestamp, 0, FREENECT_VIDEO_RGB);
        freenect_sync_get_depth(&depth, &timestamp, 0, FREENECT_DEPTH_11BIT);
        
        mFfusion.update(depth);
        const float* d = mFfusion.getMeasurement()->getDepthHost();
        
        glBindTexture(GL_TEXTURE_2D, mTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_LUMINANCE, GL_FLOAT, d);
        glBegin(GL_QUADS);
            glTexCoord2d(1.0, 0.0);
            glVertex3d(320.0, 240.0, 0.0);
            glTexCoord2d(1.0, 1.0);
            glVertex3d(320.0, -240.0, 0.0);
            glTexCoord2d(0.0, 1.0);
            glVertex3d(-320.0, -240.0, 0.0);
            glTexCoord2d(0.0, 0.0);
            glVertex3d(-320.0, 240.0, 0.0);
        glEnd();
    }
    
    void initGl(int width, int height)
    {
        DemoBase::initGl(width, height);
        
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &mTexture);
        glBindTexture(GL_TEXTURE_2D, mTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, 1, 640, 480, 0, GL_LUMINANCE, GL_FLOAT, 0);
    }
    
    void keyboardPressEvent(unsigned char key, int x, int y)
    {
        if(key == 27)
            freenect_sync_stop();
        
        DemoBase::keyboardPressEvent(key, x, y);
    }
};

int main(int argc, char** argv)
{
    Viewer(640, 480).run(&argc, argv);
    return 0;
}
