
#include "DemoBase.h"

#include "OrbitCamera.h"

#include <algorithm>
#include <stdexcept>
#include <cstdlib>

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

DemoBase* DemoBase::instance = 0;

// Trampoline functions.
inline void DemoBase::display_base()
{
    DemoBase::instance->displayBase();
}

inline void DemoBase::resize_event(int width, int height)
{
    DemoBase::instance->resizeEvent(width, height);
}

inline void DemoBase::keyboard_press_event(unsigned char key, int x, int y)
{
    DemoBase::instance->keyboardPressEvent(key, x, y);
}

inline void DemoBase::keyboard_up_event(unsigned char key, int x, int y)
{
    DemoBase::instance->keyboardUpEvent(key, x, y);
}

inline void DemoBase::mouse_button_event(int button, int state, int x, int y)
{
    DemoBase::instance->mouseButtonEvent(button, state, x, y);
}

inline void DemoBase::mouse_moved_event(int x, int y)
{
    DemoBase::instance->mouseMovedEvent(x, y);
}

DemoBase::DemoBase(int width, int height)
    : mCameraState(NONE), mWindow(0), mWidth(width), mHeight(height)
{
    if(instance != 0)
        throw std::runtime_error("There is already an instance of DemoBase.");
    
    instance = this;
    mCamera = new OrbitCamera();
    std::fill(mKeyState, mKeyState+256, false);
    std::fill(mMousePosition, mMousePosition+2, 0);
}

DemoBase::~DemoBase()
{
    delete mCamera;
}

void DemoBase::run(int* argcp, char** argv)
{
    glutInit(argcp, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
    glutInitWindowSize(mWidth, mHeight);
    
    mWindow = glutCreateWindow("FreenectFusion");
    
    glutDisplayFunc(DemoBase::display_base);
    glutIdleFunc(DemoBase::display_base);
    glutReshapeFunc(DemoBase::resize_event);
    glutKeyboardFunc(DemoBase::keyboard_press_event);
    glutKeyboardUpFunc(DemoBase::keyboard_up_event);
    glutMouseFunc(DemoBase::mouse_button_event);
    glutMotionFunc(DemoBase::mouse_moved_event);
    
    initGl(mWindow, mHeight);
    
    glutMainLoop();
}

void DemoBase::initGl(int width, int height)
{
    glClearColor(0.2, 0.2, 0.2, 0.0);
    glEnable(GL_DEPTH_TEST);
    resizeEvent(width, height);
}

void DemoBase::resizeEvent(int width, int height)
{
    double aspect = static_cast<double>(width)/static_cast<double>(height);
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, aspect, 50.0, 8000.0);
    glMatrixMode(GL_MODELVIEW);
    
    mWidth = width;
    mHeight = height;
}

void DemoBase::displayBase()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    double params[9];
    
    // Prepare the model-view matrix.
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    mCamera->getGluLookAtParameters(params);
    gluLookAt(params[0], params[1], params[2], 
              params[3], params[4], params[5],
              params[6], params[7], params[8]);
    display();
    glutSwapBuffers();      
}

void DemoBase::keyboardPressEvent(unsigned char key, int x, int y)
{
    if(key == 27)
    {
        glutDestroyWindow(mWindow);
        std::exit(0);
    }
    
    mKeyState[key] = true;
}

void DemoBase::keyboardUpEvent(unsigned char key, int x, int y)
{
    mKeyState[key] = false;
}

void DemoBase::mouseButtonEvent(int button, int state, int x, int y)
{
    mMousePosition[0] = x; mMousePosition[1] = y;
    
    if(state == GLUT_UP)
    {
        mCameraState = NONE;
        return;
    }
    
    switch(button)
    {
    case GLUT_LEFT_BUTTON:
        mCameraState = ROTATION;
        break;
    case GLUT_RIGHT_BUTTON:
        mCameraState = ZOOM;
        break;
    case GLUT_MIDDLE_BUTTON:
        mCameraState = TRANSLATION;
        break;
    default:;
    }
}

void DemoBase::mouseMovedEvent(int x, int y)
{
    double offx = x-mMousePosition[0];
    double offy = y-mMousePosition[1];
    
    switch(mCameraState)
    {
    case ROTATION:
        mCamera->rotate(offx/100.0, offy/100.0);
        break;
    case TRANSLATION:
        mCamera->translate(offx*3, -offy*3);
        break;
    case ZOOM:
        mCamera->zoom(offy);
        break;
    default:;
    }
    
    mMousePosition[0] = x; mMousePosition[1] = y;
}
