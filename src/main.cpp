
#include "DemoBase.h"
#include "OrbitCamera.h"
#include "FreenectFusion.h"
#include "MarchingCubes.h"

#include "glheaders.h"

#include <libfreenect_sync.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cerrno>
#include <iterator>
#include <cstring>

void read_calib_file(float* Krgb, float* Kdepth, float* T, const std::string& filename)
{
    std::ifstream ifs(filename.c_str());
    
    if(!ifs)
    {
        std::string err("Could not open \"");
        err += filename + "\": " + std::strerror(errno);
        throw std::runtime_error(err);
    }
    
    std::fill(Krgb, Krgb+9, 0.0);
    std::fill(Kdepth, Kdepth+9, 0.0);
    std::fill(T, T+16, 0.0);
    Krgb[8] = 1.0;
    Kdepth[8] = 1.0;
    T[15] = 1.0;
    
    std::string line;
    int i = 0;
    while(std::getline(ifs, line))
    {
        size_t pos = line.find_first_not_of(' ');
        if(pos!=std::string::npos && line[pos] == '#')
            continue;
        
        std::istringstream iss(line);
        switch(i)
        {
        case 0:
            iss >> Krgb[0] >> Krgb[4] >> Krgb[2] >> Krgb[5];
            break;
        case 1:
            iss >> Kdepth[0] >> Kdepth[4] >> Kdepth[2] >> Kdepth[5];
        case 2:
            iss >> T[0] >> T[1] >> T[2] >> T[4] >> T[5] >> T[6]
                >> T[8] >> T[9] >> T[10];
            break;
        case 3:
            iss >> T[3] >> T[7] >> T[11];
            break;
        }
        
        ++i;
    }
    
    ifs.close();
}

static void transposeTransform(float* res, const float* T)
{
    res[0] = T[0]; res[1] = T[4]; res[2] = T[8]; res[3] = T[12];
    res[4] = T[1]; res[5] = T[5]; res[6] = T[9]; res[7] = T[13];
    res[8] = T[2]; res[9] = T[6]; res[10] = T[10]; res[11] = T[14];
    res[12] = T[3]; res[13] = T[7]; res[14] = T[11]; res[15] = T[15];
}

class Viewer : public DemoBase
{
private:
    GLuint mTexture;
    FreenectFusion* mFfusion;
    VolumeMeasurement* mRenderer;
    float Krgb[9], Kdepth[9], T[16];
    
    bool mDrawFlags[3];
    
    void drawBoundingBox()
    {
        VolumeFusion* volume = mFfusion->getVolume();
        const float* bbox = volume->getBoundingBox();
        const float* p1 = bbox; const float* p2 = &bbox[3];
        glColor3d(1,1,1);
        glBegin(GL_LINE_LOOP);
            glVertex3f(p1[0], p1[1], p1[2]);
            glVertex3f(p2[0], p1[1], p1[2]);
            glVertex3f(p2[0], p2[1], p1[2]);
            glVertex3f(p1[0], p2[1], p1[2]);
        glEnd();
        glBegin(GL_LINE_LOOP);
            glVertex3f(p1[0], p1[1], p2[2]);
            glVertex3f(p2[0], p1[1], p2[2]);
            glVertex3f(p2[0], p2[1], p2[2]);
            glVertex3f(p1[0], p2[1], p2[2]);
        glEnd();
        glBegin(GL_LINES);
            glVertex3d(p1[0], p1[1], p1[2]);
            glVertex3d(p1[0], p1[1], p2[2]);
            glVertex3d(p2[0], p1[1], p1[2]);
            glVertex3d(p2[0], p1[1], p2[2]);
            glVertex3d(p2[0], p2[1], p1[2]);
            glVertex3d(p2[0], p2[1], p2[2]);
            glVertex3d(p1[0], p2[1], p1[2]);
            glVertex3d(p1[0], p2[1], p2[2]);
        glEnd();
    }
    
    void drawSensor()
    {
        float aux[16];
        glPushMatrix();
        transposeTransform(aux, mFfusion->getLocation());
        glMultMatrixf(aux);
        glPointSize(5);
        glBegin(GL_POINTS);
        glColor3d(1.0,1.0,1.0);
        glVertex3d(0., 0., 0.);
        glColor3d(1., 0., 0.);
        glVertex3d(100.0, 0, 0);
        glColor3d(0., 1., 0.);
        glVertex3d(0, 100.0, 0);
        glColor3d(0., 0., 1.);
        glVertex3d(0, 0, 100.0);
        glEnd();
        glPopMatrix();
    }
    
public:
    Viewer(int width, int height, const std::string calib_filename)
        : DemoBase(width, height), mFfusion(0)
    {
        read_calib_file(Krgb, Kdepth, T, calib_filename);
        mDrawFlags[0] = true;
        mDrawFlags[1] = true;
        mDrawFlags[2] = false;
    }
    
    ~Viewer()
    {
        delete mFfusion;
        delete mRenderer;
    }
    
protected:
    void display()
    {
        void* image = 0;
        void* depth = 0;
        float aux[16];
        uint32_t timestamp;
        freenect_sync_get_video(&image, &timestamp, 0, FREENECT_VIDEO_RGB);
        freenect_sync_get_depth(&depth, &timestamp, 0, FREENECT_DEPTH_11BIT);
        
        mFfusion->update(depth);
        
        glDisable(GL_TEXTURE_2D);
        glPointSize(1);
        
        glMatrixMode(GL_MODELVIEW);
        
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        
        glRotated(180, 0, 0, 1);
        
        if(mDrawFlags[0])
        {
            glPushMatrix();
            const double* t = getCamera()->getTransform();
            float t2[16];
            std::copy(t, t+16, t2);
            mRenderer->measure(*mFfusion->getVolume(), t2);
            
            glBindBuffer(GL_ARRAY_BUFFER, mRenderer->getGLVertexBuffer());
            glVertexPointer(3, GL_FLOAT, 3*sizeof(float), 0);
            glBindBuffer(GL_ARRAY_BUFFER, mRenderer->getGLNormalBuffer());
            glColorPointer(3, GL_FLOAT, 3*sizeof(float), 0);
            glDrawArrays(GL_POINTS, 0, mRenderer->getNumVertices());
            glPopMatrix();
        }
        
        if(mDrawFlags[1])
        {
            glBindBuffer(GL_ARRAY_BUFFER, mFfusion->getVolumeMeasurement()->getGLVertexBuffer());
            glVertexPointer(3, GL_FLOAT, 12, 0);
            glBindBuffer(GL_ARRAY_BUFFER, mFfusion->getVolumeMeasurement()->getGLNormalBuffer());
            glColorPointer(3, GL_FLOAT, 12, 0);
            glDrawArrays(GL_POINTS, 0, 640*480);
        }
        
        if(mDrawFlags[2])
        {
            glPushMatrix();
            transposeTransform(aux, mFfusion->getLocation());
            glMultMatrixf(aux);
            glBindBuffer(GL_ARRAY_BUFFER, mFfusion->getMeasurement()->getGLVertexBuffer(1));
            glVertexPointer(3, GL_FLOAT, 12, 0);
            glBindBuffer(GL_ARRAY_BUFFER, mFfusion->getMeasurement()->getGLNormalBuffer(1));
            glColorPointer(3, GL_FLOAT, 12, 0);
            glDrawArrays(GL_POINTS, 0, 640*480);
            glPopMatrix();
        }
        
        drawBoundingBox();
        drawSensor();
        
        /*if(mDrawFlags[0])
        {
            MarchingCubes* mc = mFfusion->getMarchingCubes();
            glBindBuffer(GL_ARRAY_BUFFER, mc->getGLVertexBuffer());
            glVertexPointer(4, GL_FLOAT, 4*sizeof(float), 0);
            glBindBuffer(GL_ARRAY_BUFFER, mc->getGLNormalBuffer());
            glColorPointer(4, GL_FLOAT, 4*sizeof(float), 0);
            glDrawArrays(GL_TRIANGLES, 0, mc->getActiveVertices());
        }*/
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
        
        mFfusion = new FreenectFusion(640, 480, Kdepth, Krgb);
        
        const float K[9] = {450, 0, width/2.f, 0, 450, height/2.f, 0, 0, 1.f};
        mRenderer = new VolumeMeasurement(width, height, K);
    }
    
    void keyboardPressEvent(unsigned char key, int x, int y)
    {
        if(key == 27)
            freenect_sync_stop();
        if(key == '0')
            mDrawFlags[0] ^= true;
        if(key == '1')
            mDrawFlags[1] ^= true;
        if(key == '2')
            mDrawFlags[2] ^= true;
        if(key == 't')
            mFfusion->toggleTracking();
        if(key == 'u')
            mFfusion->toggleUpdate();
        
        DemoBase::keyboardPressEvent(key, x, y);
    }
};

void print_usage(const char* execname)
{
    std::cout << "Usage:\n\n\t" << execname << " CALIBRATIONFILE\n" << std::endl;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }
    
    char* calib_filename = argv[1];
    Viewer(640, 480, calib_filename).run(&argc, argv);
    return 0;
}
