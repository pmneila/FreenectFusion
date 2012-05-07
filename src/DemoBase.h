
#ifndef _DEMOBASE_H
#define _DEMOBASE_H

class OrbitCamera;

class DemoBase
{
private:
    OrbitCamera* orbitCamera;
    
public:
    DemoBase();
    virtual ~DemoBase();
    virtual void run();
    
protected:
    virtual void initGl();
    virtual void resizeEvent();
    virtual void keyboardPressEvent(int key, int x, int y);
    virtual void keyboardUpEvent(int key, int x, int y);
    virtual void mouseButtonEvent(int button, int state, int x, int y);
    virtual void mouseMovedEvent(int x, int y);
    virtual void display() = 0;
    
private:
    void displayBase();
};

#endif // _DEMOBASE_H
