
#include "DemoBase.h"
#include "FreenectFusion.h"

#include "glheaders.h"

#include <libfreenect_sync.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cerrno>
#include <iterator>
#include <cstring>

void read_calib_file(double* Krgb, double* Kdepth, double* T, const std::string& filename)
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

class Viewer : public DemoBase
{
private:
    GLuint mTexture;
    FreenectFusion* mFfusion;
    double Krgb[9], Kdepth[9], T[16];
    
public:
    Viewer(int width, int height, const std::string calib_filename)
        : DemoBase(width, height), mFfusion(0)
    {
        read_calib_file(Krgb, Kdepth, T, calib_filename);
    }
    
    ~Viewer()
    {
        delete mFfusion;
    }
    
protected:
    void display()
    {
        void* image = 0;
        void* depth = 0;
        uint32_t timestamp;
        freenect_sync_get_video(&image, &timestamp, 0, FREENECT_VIDEO_RGB);
        freenect_sync_get_depth(&depth, &timestamp, 0, FREENECT_DEPTH_11BIT);
        
        mFfusion->update(depth);
        
        glDisable(GL_TEXTURE_2D);
        glPointSize(1);
        
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, mFfusion->getVolumeMeasurement()->getGLVertexBuffer());
        glVertexPointer(3, GL_FLOAT, 12, 0);
        glBindBuffer(GL_ARRAY_BUFFER, mFfusion->getVolumeMeasurement()->getGLNormalBuffer());
        glColorPointer(3, GL_FLOAT, 12, 0);
        glDrawArrays(GL_POINTS, 0, 640*480);
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
    }
    
    void keyboardPressEvent(unsigned char key, int x, int y)
    {
        if(key == 27)
            freenect_sync_stop();
        
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
