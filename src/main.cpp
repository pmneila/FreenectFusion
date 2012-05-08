
#include "DemoBase.h"

class Viewer : public DemoBase
{
public:
    Viewer(int width, int height)
        : DemoBase(width, height)
    {}
    
protected:
    void display()
    {
        
    }
    
    void initGl(int width, int height)
    {
        DemoBase::initGl(width, height);
    }
};

int main(int argc, char** argv)
{
    Viewer(640, 480).run(&argc, argv);
    return 0;
}
