
#ifndef _DEMOBASE_H
#define _DEMOBASE_H

class OrbitCamera;

typedef enum
{
    NONE=0,
    ROTATION=1,
    TRANSLATION=2,
    ZOOM=3
} CameraState;

class DemoBase
{
private:
    OrbitCamera* mCamera;
    CameraState mCameraState;
    int mWindow;
    int mWidth, mHeight;
    bool mKeyState[256];
    int mMousePosition[2];
    
public:
    DemoBase(int width, int height);
    virtual ~DemoBase();
    virtual void run(int* argcp, char** argv);
    
    static DemoBase* instance;
    
protected:
    virtual void initGl(int width, int height);
    virtual void resizeEvent(int width, int height);
    virtual void keyboardPressEvent(unsigned char key, int x, int y);
    virtual void keyboardUpEvent(unsigned char key, int x, int y);
    virtual void mouseButtonEvent(int button, int state, int x, int y);
    virtual void mouseMovedEvent(int x, int y);
    virtual void display() = 0;
    
private:
    void displayBase();
    
    // Trampoline functions.
    static void display_base();
    static void resize_event(int width, int height);
    static void keyboard_press_event(unsigned char key, int x, int y);
    static void keyboard_up_event(unsigned char key, int x, int y);
    static void mouse_button_event(int button, int state, int x, int y);
    static void mouse_moved_event(int x, int y);
};

#endif // _DEMOBASE_H
