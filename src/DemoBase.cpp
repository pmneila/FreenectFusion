
#include "DemoBase.h"

#include "OrbitCamera.h"

DemoBase::DemoBase()
{
    orbitCamera = new OrbitCamera();
}

DemoBase::~DemoBase()
{
    delete orbitCamera;
}
