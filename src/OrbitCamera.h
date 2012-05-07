
#ifndef _ORBITCAMERA_H
#define _ORBITCAMERA_H

class OrbitCamera
{
private:
    double center[3];
    double radius;
    double longitude, latitude;
    mutable double position[2];
    
public:
    
    OrbitCamera();
    virtual ~OrbitCamera();
    
    void translate(double dx, double dy);
    void rotate(double dlongitude, double dlatitude);
    void zoom(double dy);
    
    const double* getPosition() const;
    const double* getVector() const;
    void getGluLookAtParameters(double* params) const;
};

#endif // _ORBITCAMERA_H
